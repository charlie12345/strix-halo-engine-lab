# Prellenado en pareja y continuación en el rango 0

El traspaso en pareja es explícito: establezca `STRIX_QWEN_PF_PIPE=1` y
`STRIX_QWEN_PF_PIPE_DECODE=1` y, a continuación, lance un proceso por rango con
`--rank 0|1`. El transporte predeterminado es `usb4gpu` con el router por
tamaño; `--transport usb4auto` (o `usb4-auto`) restaura host. Los gates nativos
bit a bit/TTFT del seam directo están pendientes; las mediciones siguientes
usaron host. Véase la [decisión de transporte](../../coord/transport-routing.md). Utilice binarios, recursos del modelo,
identificadores de la petición, opciones numéricas y capacidad idénticos. El
fragmento predeterminado de la pareja es de 4,096 filas cuando se admite. Guardar
la sesión después de un prellenado en dos nodos requiere una longitud de petición
que sea múltiplo del tamaño del fragmento. Medido en dos nodos por USB4 (comprobación de la versión fusionada, un par nuevo por ejecución): tokens,
logits completos y estado final iguales a la ejecución en un nodo a 32K, 128K y con un prompt impar de
32,771 tokens. Prefill 32K **12.1-12.3 s (~2,660-2,710 tok/s)** frente a 19.2 s en un nodo; 128K
**46.5-46.9 s (~2,800 tok/s)** frente a 79.8 s.

El resolvedor ejecuta las peticiones cortas íntegramente en el rango 0, sin conexión
ni traspaso. En otro caso, canaliza un prefijo legal, restaura la ejecución del
modelo completo en el rango 0 y consume el sufijo antes de la generación. Los
prefijos simples requieren mosaicos completos o una cola de al menos 16 filas que
sea múltiplo de ocho. MTP respeta además su escalera real de propietarios; su
continuación utiliza la frontera existente del predictor y de la sesión. El rango 1
termina tras su trabajo. Un conjunto cargado de costes por capa y de transferencia
compara las estimaciones en serie y en canalización, incluido el traspaso. Sin él,
el cruce provisional es de 28,672 filas: el ajuste disponible de ahorros de extremo
a extremo en 32K/128K cruza en 24,907, redondeado hacia arriba sobre la rejilla
medida de 4,096 filas. Vuelva a medir este umbral con fragmentos de 4,096 filas y el
nuevo traspaso; es una estimación, no una afirmación de velocidad para contextos
cortos.

El empaquetado de la costura protege con una barrera la instantánea D2D antes de
que flujos independientes puedan reutilizar las arenas de origen. Los archivos de
diagnóstico utilizan una copia propia de los bytes de preparación autenticados. La
versión 2 del protocolo de estado final utiliza dos ranuras fijadas de 16 MiB, un
flujo de copia separado, CRC32C por pieza y una única confirmación ACK tras todas
las instalaciones H2D. Ambos rangos deben utilizar esta versión del protocolo. El
coste de memoria es de 32 MiB de memoria del host fijada por cada traspaso activo.

## Equilibrio de etapas, copias de la costura y línea temporal de la pareja

`STRIX_PAIR_TIMING=1` imprime una línea `pair-timing` por fragmento y rango y una suma por
rango (el `strix::dist::PipelineTimeline` compartido): duración del fragmento, espera para
adquirir la costura, barrera del productor, empaquetado/carga de la costura y, en el hilo de
progreso, espera de D2H, CRC, cable, espera de ACK, espera de cabecera y espera de H2D; el
traspaso de estado es la última fila. Son relojes del host de llamadas bloqueantes; la
barrera del productor es donde el host espera el trabajo de GPU del fragmento. Desactivado,
no añade trabajo.

Medido con ella (fragmentos de 4,096 filas, 32K, dos nodos, ruta de host USB4): por fragmento
la costura cuesta 4-5 ms de CRC, ~18 ms de cable para 105 MB y ~4 ms de espera de ACK; el
rango 0 nunca espera una ranura de trama. La costura no es el límite. Lo es la etapa A (capas
antes del corte), que además ejecuta el preludio de incrustación/PLE, por lo que con el
reparto par 24/24 es la etapa más lenta. El corte predeterminado es ahora **23**
(`cut-source=measured-default`); `STRIX_QWEN_PF_PIPE_CUT=24` restaura el reparto par. Barrido
de cortes (logits iguales bit a bit al prellenado en un nodo en todos los brazos): 32K 24:
12.44/12.51/12.50 s, 23: 12.08, 22: 12.31, 21: 12.63/12.65, 20: 13.00; 128K 24: 49.36/50.15 s,
23: 47.55/47.53.

El empaquetado (rango 0) y la dispersión (rango 1) de la costura movían las filas de cada
fragmento con 3 x 4,096 copias pequeñas de dispositivo (~27 ms por fragmento en cada rango).
Ahora usan una copia con paso por componente (los mismos bytes).

La distancia restante hasta 2x es estructural en una canalización de dos etapas por capas:
el tiempo es aproximadamente (tiempo en un nodo con el tamaño de fragmento de la pareja) / 2 x
(1 + fragmento / petición), más el traspaso de estado (0.11 s a 32K, 0.4 s a 128K). Otro tamaño
para el primer o el último fragmento no lo acorta (el fragmento mayor fija el llenado más el
vaciado), y fragmentos uniformes más pequeños pierden más eficiencia por nodo de la que
ahorran (un nodo a 32K: mosaicos de 4,096 filas 20.6-21.1 s frente a 19.0-20.0 s con 16,384,
todos iguales bit a bit).

## Referencia MTP ligada a la petición y reproducción

Conserve los argumentos calificados de recursos, procedencia y ventana en el archivo
privado `assets/mtp-qualified.argv`, exactamente un argumento por línea. Incluye
`--mtp` y todas las fijaciones necesarias de archivos auxiliares y de destino, la
política y los presupuestos de dispositivo, pero excluye el modelo, la petición, los
indicadores de pareja, las opciones de la referencia y los dos presupuestos de
archivos del host indicados a continuación. Conserve sus opciones numéricas en todas
las órdenes. Aquí no se publica ningún valor de recursos privados. Estas matrices de
Bash describen el lanzamiento completo:

```bash
mapfile -t MTP < assets/mtp-qualified.argv
COMMON=(MODEL.gguf --prompt-ids assets/prompt-32768.ids --context 131072 --generate 128
        "${MTP[@]}" --mtp-golden-max-bytes 34359738368 --mtp-replay-host-bytes 35433480192)
# Captura en un solo nodo; el destino no debe existir previamente.
env -u STRIX_QWEN_PF_PIPE -u STRIX_QWEN_PF_PIPE_DECODE -u STRIX_QWEN_PF_LAYER_RANGE \
  strix-hip-decode-token "${COMMON[@]}" --mtp-golden-dump assets/prompt-32768.golden
SHA=$(sha256sum assets/prompt-32768.golden); SHA=${SHA%% *}
REPLAY=("${COMMON[@]}" --mtp-golden assets/prompt-32768.golden --mtp-golden-sha256 "$SHA")
env -u STRIX_QWEN_PF_PIPE -u STRIX_QWEN_PF_PIPE_DECODE -u STRIX_QWEN_PF_LAYER_RANGE \
  strix-hip-decode-token "${REPLAY[@]}"
# Copie antes exactamente los mismos recursos de referencia/argv/petición/modelo al otro rango.
# Ejecute una vez por nodo, con R igual a 0 y 1 respectivamente:
R=0
STRIX_QWEN_PF_PIPE=1 STRIX_QWEN_PF_PIPE_DECODE=1 STRIX_QWEN_PF_PIPE_CHUNK=4096 \
  strix-hip-decode-token "${REPLAY[@]}" --rank "$R" --transport usb4auto
```

La referencia de 32K contiene 32,767 pares adyacentes de la petición, unos 33.89 GB
para la geometría 2560x4/248320. Reserve la memoria completa de reproducción además
de la memoria del dispositivo y del modelo. Cambie la petición a 1,193 tokens solo
después de capturar y fijar su propia referencia; una referencia de 32K no puede
validar esa petición. Las referencias generadas por el propio motor son evidencia de
calibración, no una calificación independiente de la fidelidad del modelo. Se
rechaza el volcado de una referencia en pareja; la captura se hace en un solo nodo y
la reproducción en pareja.
