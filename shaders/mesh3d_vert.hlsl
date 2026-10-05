struct VertexInput {
    float3 position : POSITION;
    float3 color : COLOR0;
};

struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR0;
};
[[vk::binding(0, 0)]] cbuffer CameraData {
    float4x4 view_projection;
};

VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = mul(view_projection, float4(input.position, 1.0));
    output.color = input.color;
    return output;
}
