# Almacenamiento KV y división por tokens en GLM

Las solicitudes con contexto largo necesitan memoria para cada token cuyo estado de atención deba seguir disponible. El estado de claves y valores (KV) crece con la solicitud, junto con los pesos del modelo, el espacio de trabajo y los demás estados de la solicitud. Por tanto, la cantidad de tokens que cabe depende de la organización de la caché de cada motor y de la admisión de memoria al inicio.

## Cómo almacenan el estado de atención los motores

La comparación siguiente describe una solicitud en un par de dos nodos con APU de 128 GB, conectados por USB4, con contexto de 8K salvo que la fila indique otro valor. Las cifras por token describen la contabilidad de la caché para este par; no son una afirmación sobre el contexto máximo.

| Motor | Estado por token y nodo | Organización |
|---|---:|---|
| GLM | 17,737 B | El estado latente de atención multi-cabeza (MLA) de las 11 capas de atención dispersa (DSA) se replica en ambos nodos con la organización exacta predeterminada. |
| MiniMax | 76,800 B | El estado de atención ya se divide entre los dos nodos por cabezas de atención. |
| DeepSeek V4.1 | Aproximadamente 894 B | El estado comprimido se replica en ambos nodos. |
| Qwen híbrido | Varía por capa y formato | El estado se divide por capa solo durante el prefill de dos nodos; después, el rango 0 conserva el estado completo. |

La opción de esta guía cambia la propiedad por token del estado latente DSA de GLM. No cambia la organización KV de MiniMax, DeepSeek V4.1 ni Qwen.

## Qué hace la división por tokens de GLM

Con la división por tokens, cada nodo conserva el estado latente de sus propios bloques de tokens. Ambos nodos intercambian claves de puntuación para que la selección global top-k siga siendo igual a la de la ruta replicada. Cada nodo calcula su parcial de atención sobre los tokens que posee; los parciales se combinan en orden fijo de rangos y en FP32.

El cambio en el orden de reducción crea un perfil numérico FAST independiente; no es bit a bit igual al perfil exacto replicado predeterminado. Su identidad es:

`glm-dsa-kvshard-rank01-fp32-v3-declaredexp-owned-tree256-fma-rankmerge-FAST`

En el par de dos nodos con APU de 128 GB por USB4, con una solicitud y contexto de 8K, los vectores de tokens generados coincidieron con la ejecución exacta replicada en 128/128 posiciones comprobadas. Con contexto de 64K en el mismo par y una solicitud, coincidieron en 32/32 posiciones comprobadas. Esta coincidencia de tokens no hace que los logits sean iguales bit a bit: las diferencias de logits son del orden del reordenamiento y aumentan con la profundidad. Para calificar una ejecución, compare la diferencia absoluta máxima de los logits.

## Cuándo usarlo y cómo se selecciona

El valor predeterminado `STRIX_GLM_KV_SHARD=auto` conserva el estado replicado cuando supera la admisión de inicio. Selecciona la división cuando la organización replicada no supera la admisión y la dividida sí. El inicio informa de la organización efectiva y del motivo de admisión. Use `auto` si desea admitir el contexto más largo posible sin elegir manualmente la organización.

Establezca `STRIX_GLM_KV_SHARD=1` para forzar la división o `STRIX_GLM_KV_SHARD=0` para forzar la organización replicada. Use el mismo valor en ambos rangos. El inicio falla si la división forzada no cabe; también falla si la replicación forzada no cabe.

La configuración calificada actual admite una solicitud (`slots 1`), con `STRIX_GLM_PF_PIPELINE=1`, prefill por chunks y sin MTP ni otra especulación. La división no se elige para aumentar la velocidad: reduce el uso de caché por nodo a cambio de más intercambio y trabajo de reducción. La identidad indicada corresponde a la ejecución FAST dividida. El modo exacto replicado predeterminado sigue siendo el valor bit a bit predeterminado.

## Capacidad y velocidad medidas

Todas las cifras siguientes se midieron para una solicitud en un par de dos nodos con APU de 128 GB por USB4. «Exacto» significa la organización replicada predeterminada bit a bit; «FAST» significa el perfil dividido indicado arriba. El contexto es el contexto del prompt y la velocidad de decode se expresa en milisegundos por token generado.

| Medición | Exacto replicado | FAST dividido por tokens |
|---|---:|---:|
| Contexto máximo admitido | 1.45M tokens, con espacio de trabajo de prefill acotado | Aproximadamente 2.85M tokens; se ejecutó con 2.85M y el inicio rechazó 2.9M |
| Decode con contexto de 8K | 39.7 ms/token | 41.5 ms/token |
| Decode con contexto de 2.0M–2.85M | — | 43.6–45.7 ms/token |
| Prefill con contexto de 8K | 17.1 s | 18.2 s |
| Prefill con contexto de 64K | 145.1 s | 155.9 s |

Estos son máximos admitidos y tiempos medidos para este par y esta configuración, no garantías para otro hardware o carga. Use la división cuando el estado replicado no admita el contexto que necesita y espere un decode más lento cuando quepan ambas organizaciones.

## Cómo verificar el perfil seleccionado

Cada rango imprime la decisión de admisión del inicio y la organización efectiva. Si la decisión es automática, revise las líneas de inicio de ambos rangos para ver qué organización se admitió y por qué. La línea de identidad efectiva tiene esta forma:

```text
rank R kv-shard effective=1 identity=glm-dsa-kvshard-rank01-fp32-v3-declaredexp-owned-tree256-fma-rankmerge-FAST
```

Para la ejecución replicada, la identidad efectiva es `replicated`. El recibo de ejecución también incluye la identidad numérica. Conserve la salida de inicio y el recibo junto con el prompt, el contexto y la configuración de la ejecución.

Para comparar ambas organizaciones, ejecute dos veces el mismo prompt tokenizado, los mismos activos del modelo, los mismos ajustes de solicitud y el mismo contexto en el mismo par: una vez con `STRIX_GLM_KV_SHARD=0` y otra con `STRIX_GLM_KV_SHARD=1` (o `auto` cuando la admisión seleccione la división). Compare los vectores completos de tokens generados y la diferencia absoluta máxima entre los logits correspondientes. Que coincidan los tokens aporta evidencia útil para ese prompt y esa longitud; no demuestra equivalencia de logits bit a bit.

Para el diagnóstico por capa, asigne `STRIX_GLM_KVSHARD_DUMP` a un directorio de salida antes de iniciar ambos rangos. El volcado opcional registra los IDs de tokens seleccionados, las puntuaciones de los pools y las filas de salida de atención de la ruta DSA dividida. Úselo para localizar dónde divergen por primera vez la selección o los valores de atención; es información de diagnóstico, no sustituye la comparación completa de ejecuciones.

## Costes y límites

- La división reduce el almacenamiento latente DSA replicado porque cada rango conserva solo sus propios bloques de tokens. Añade intercambio de claves de puntuación, atención parcial en cada rango y una combinación FP32 ordenada.
- El perfil FAST cambia el orden de suma. Sus logits difieren de los del perfil exacto replicado, aunque los tokens generados coincidan en los casos comprobados.
- En este par, las mediciones de decode y prefill a 8K son más lentas con la división. Prefiera la organización exacta replicada cuando admita la solicitud y necesite identidad bit a bit con el valor predeterminado.
- El alcance calificado actual es una solicitud, prefill por pipeline y ausencia de MTP o especulación. No deduzca de estas cifras el rendimiento con varias solicitudes, especulación, otro hardware o prefill por encima de 64K.
- La selección automática es una decisión de admisión, no una heurística de velocidad. Si ninguna organización supera la admisión de inicio, se rechaza la solicitud en vez de superar silenciosamente la capacidad admitida.
