# Grafo y plan descriptivos de MiniMax

El adaptador de MiniMax valida `minimax-m3.1-flash.toml` con el contrato compilado
de descripción del modelo antes de construir un `ModelGraph` común. Se trata de
herramientas de descripción y diagnóstico; el motor existente ejecuta su propio
calendario. No se sustituye ninguna identidad numérica, de caché, de estado guardado
ni de ejecución compilada.

El grafo registra el tronco MSA completo de 60 capas, la semántica de RMSNorm de
Gemma por cabeza y de rotación, el indexador de bloques, las tres cachés de
secuencia, el prefijo denso, los expertos enrutados y los compartidos sin escalar,
las aristas residuales, la norma final y la cabeza de vocabulario. La posibilidad de
particionar las cabezas y la dimensión intermedia del MLP es semántica del modelo;
los rangos concretos de cada rango TP2 y el relleno del vocabulario pertenecen al
plan. Ambos rangos poseen porciones intermedias de los 128 identificadores de
expertos enrutados. No poseen rangos disjuntos de identificadores de expertos.

Los recursos de borrador configurados aparecen como un adjunto de borrador opaco.
Sus bloques internos, tomas, verificación y calendario de aceptación siguen siendo
locales del motor. La descripción de origen no declara ningún bloque MTP, por lo que
el adaptador no infiere un adjunto MTP. La presencia de un adjunto describe la
configuración, no una carga correcta de los recursos.

Coloque la descripción del modelo del repositorio junto a los recursos del modelo y,
a continuación, inspecciónelo con la forma de lanzamiento habitual en cualquiera de
los rangos:

```sh
strix-minimax-rank MODEL_DIR 0 usb4-auto OUTPUT_DIR --print-model-graph
strix-minimax-rank MODEL_DIR 1 usb4-auto OUTPUT_DIR --print-plan
strix-minimax-serve MODEL_DIR --rank 0 --transport usb4-auto --print-plan
```

`STRIX_PRINT_MODEL_GRAPH=1` y `STRIX_PRINT_EXEC_PLAN=1` son activaciones
equivalentes. La impresión termina antes del transporte, de las consultas HIP, de la
carga de pesos o del arranque del servidor. Si se solicitan ambas, el JSON del grafo
precede al JSON del plan, un registro por línea.
`STRIX_MINIMAX_MODEL_GRAPH_CHECK=1` comprueba el grafo frente al archivo auxiliar
validado y valida su plan antes del transporte; después continúa la ejecución
normal. Un archivo auxiliar ausente o discordante solo es un error para estas
activaciones; la ejecución ordinaria conserva su comportamiento actual de admisión
de la descripción del modelo.

El plan común proyecta una fila lógica de decodificación y, después, el límite de la
cabeza voraz: norma de entrada; proyecciones QKV/de índice; normas por cabeza y RoPE;
anexado de KV/índice; puntuaciones/selección de bloques y atención; proyección de
salida; residual ordenado X1; norma posterior a la atención; MLP denso o
router/enrutados/compartidos/combinación; residual ordenado X2; norma final, logits
del fragmento de vocabulario y argmax del par. Un listado topológico no especifica
la serialización real de los flujos. Las normas fusionadas en un límite de
intercambio se listan por separado según su papel matemático; no se implica ningún
número de lanzamientos.

El plan registra los valores del resolvedor puro `mm_defaults.cpp` como política
declarada, no como evidencia de que un dispositivo haya seleccionado esos núcleos.
Enumera la selección de núcleos, la fusión, el solapamiento de flujos/eventos, el
espacio temporal físico y la asignación de cachés, las esperas/publicaciones del
transporte, la planificación del prellenado, la admisión del servicio y el borrador
que no se proyectan. El espacio temporal dinámico del índice tiene extensiones
simbólicas y cero bytes no proyectados. Los demás tamaños de activación describen
una fila lógica, y los bytes de secuencia son por posición retenida, no una
estimación de admisión de memoria. Esta proyección no califica ningún nodo para la
captura de grafos. Las pruebas en el host no pueden establecer la ejecución en el
dispositivo ni la igualdad de bits.

Para un primer módulo RMSNorm compartido, conserve el orden de evaluación
`(x*r)*(1+w)` de MiniMax, el árbol de reducción en FP32, el redondeo del épsilon y
la ubicación en el límite de intercambio. La equivalencia matemática por sí sola no
basta para sustituir una implementación calificada.
