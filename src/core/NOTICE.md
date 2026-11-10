Shared tiled core and stored INT4 support

The tiled core, its operand/contract interfaces, IU4 pack builder and support
headers are adopted from the project strix-trellis library, policy interface v1.
The small-row W4A16 decoder is adopted from that library and extended here with
streaming stages and shared dequantisation. Apache License 2.0 applies; see
LICENSES/strix-trellis-Apache-2.0.txt. Existing attribution is retained.

The EXL3 spelling identifies the adopted exllamav3 format/quantizer basis.
FAST_I4 is a separate lossy numerical identity. Conversion occurs only at load;
no fused requantisation or runtime weight conversion is provided.
