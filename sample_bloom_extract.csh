struct FragmentShaderInputs with
  [DescriptorSet 0, Binding 0, InputAttachment 0]
  sourceTex: SubpassInput
  [PushConstant 0]
  threshold: Float

[FragmentShader] # fsh
fragMain inputs: FragmentShaderInputs -> [Location 0] Float4 do
  let i = subpassLoad inputs.sourceTex
  let i = subpassLoad inputs.sourceTex
  if i.r >= inputs.threshold || i.g >= inputs.threshold || i.b >= inputs.threshold then i else Float4 0.0f

[VertexShader]
vertMain [VertexID] vx: Int -> [Position] Float4 = Float4(
  if vx & 0x01 == 0 then -1.0f else 1.0f,
  if vx & 0x02 == 0 then -1.0f else 1.0f,
  0.0f,
  1.0f
)
