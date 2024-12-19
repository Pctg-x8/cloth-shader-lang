
[VertexShader]
vertMain (
    [Location 0] pos: Float4,
    [Location 1] texcoord: Float4,
    [DescriptorSet 0, Binding 0] uniform viewProjectionMatrix: Float4x4,
    [DescriptorSet 1, Binding 0] uniform objectTransformMatrix: Float4x4
) -> [Position] Float4 =
    pos * objectTransformMatrix * viewProjectionMatrix

[FragmentShader]
fragMain () -> [Location 0] Float4 = Float4 1.0f
