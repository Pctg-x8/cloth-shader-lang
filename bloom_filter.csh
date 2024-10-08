# peridot basic bloom filter with dual filter

[VertexShader]
vertMain [VertexID] vid: Int -> ([Position] Float4, [Location 0] Float2) =
  let pos = Float4(vid `vpos` 0x01, vid `vpos` 0x02, 0.0, 1.0)

  (pos, (pos * 0.5 + 0.5).xy)

vpos (id: Int, bits: Int) -> Float = if id & bits == 0 then -1.0 else 1.0

struct FragmentShaderInputs with
  [Location 0] uv: Float2
  [DescriptorSet 0, Binding 0] tex: Texture2D
  [PushConstant 0] texelSize: Float2

[FragmentShader]
fragDownSample input: FragmentShaderInputs -> [Location 0] Float4 =
  let offset = Float4(-1.0, 1.0, -1.0, 1.0) * input.texelSize.xxyy
  let mut res = 4 * input.tex `sampleAt` input.uv
  res += input.tex `sampleAt` (input.uv + offset.xz)
  res += input.tex `sampleAt` (input.uv + offset.xw)
  res += input.tex `sampleAt` (input.uv + offset.yz)
  res += input.tex `sampleAt` (input.uv + offset.yw)
  res / 8.0

[FragmentShader]
fragUpSample (input: FragmentShaderInputs, [DescriptorSet 1, Binding 0] upperBloomTex: Texture2D) -> [Location 0] Float4 =
  let offset = Float4(-1.0, 1.0, -1.0, 1.0) * input.texelSize.xxyy
  let mut res = input.tex `sampleAt` (input.uv + Float2(offset.x, 0.0))
  res += input.tex `sampleAt` (input.uv + Float2(offset.y, 0.0))
  res += input.tex `sampleAt` (input.uv + Float2(0.0, offset.z))
  res += input.tex `sampleAt` (input.uv + Float2(0.0, offset.w))
  res += 2.0 * input.tex `sampleAt` (input.uv + offset.xz * 0.5)
  res += 2.0 * input.tex `sampleAt` (input.uv + offset.xw * 0.5)
  res += 2.0 * input.tex `sampleAt` (input.uv + offset.yz * 0.5)
  res += 2.0 * input.tex `sampleAt` (input.uv + offset.yw * 0.5)
  res / 12.0 + upperBloomTex `sampleAt` input.uv
