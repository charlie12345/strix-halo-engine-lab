# Prellenado en pareja y continuación en el rango 0

El traspaso en pareja es explícito: establezca `STRIX_QWEN_PF_PIPE=1` y
`STRIX_QWEN_PF_PIPE_DECODE=1` y, a continuación, lance un proceso por rango con
`--rank 0|1 --transport usb4auto`. Utilice binarios, recursos del modelo,
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
