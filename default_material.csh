struct CameraData:
  viewProjectionMatrix: Float4x4

struct PerObjectData:
  toWorldMatrix: Float4x4

struct LightData:
  incidentDir: Float3
  intensity: Float

struct VertexInput:
  [Location 0] pos: Float4
  [Location 1] normal: Float4
  [Location 2] uv: Float4

struct VaryingParameters:
  [Location 0] normal: Float4
  [Location 1] uv: Float4

normalize3 i: Float4 -> Float4 = Float4(normalize i.xyz, i.w)

[VertexShader]
vertMain (
  v: VertexInput,
  [DescriptorSet 0, Binding 0] cameraData: CameraData,
  [DescriptorSet 1, Binding 0] perObjectData: PerObjectData
) -> ([Position] Float4, VaryingParameters) = (
  v.pos * perObjectData.toWorldMatrix * cameraData.viewProjectionMatrix,
  VaryingParameters {
    normal = normalize3(transpose perObjectData.toWorldMatrix * v.normal),
    uv = v.uv
  }
)

[FragmentShader]
fragMain (v: VaryingParameters, [DescriptorSet 1, Binding 1] lightData: LightData) -> [Location 0] Float4 =
  let diffuse = ((v.normal.xyz `dot` -lightData.incidentDir) * 0.5f + 0.5f) ^^ 2.0f * lightData.intensity
  Float4(Float3 1.0f * diffuse + Float3 0.3f, 1.0f)

