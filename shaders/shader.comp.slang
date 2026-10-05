//------------------------------------------------------------------------------
// Compute shader demonstrating vertex transformation and buffer device address usage
// This shader rotates vertices at different speeds based on their index
//------------------------------------------------------------------------------

#include "shader_io.h"

// Output texture that can be both read and written
// Note: RWTexture2D is equivalent to Vulkan's storage image
[[vk::binding(0)]] 
RWTexture2D<float4> outImage;

// Push constants for frequently updated data
// Note: Using push constants is more efficient than constant buffers for small, frequently changing data
[[vk::push_constant]]  
ConstantBuffer<PushConstantCompute> pushConst;

// Declare this as a compute shader with [shader("compute")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)  // SV_DispatchThreadID gives us the global thread ID
{
    uint index = threadIdx.x;
    
    if(index >= pushConst.numVertex)
        return;
    
    // Access the vertex buffer using buffer device address
    // Note: This is using Vulkan's VK_KHR_buffer_device_address extension
    // Cast the uint64 address to a typed pointer for direct GPU buffer access
    Vertex* vertices = (Vertex*)pushConst.bufferAddress;
    
    float angle = pushConst.rotationAngle * sin((index + 1) * 2.3f);
    
    float3 vertex = vertices[index].position;
    
    // Construct a 3D rotation matrix around the Z-axis
    float cosAngle = cos(angle);
    float sinAngle = sin(angle);
    float3x3 rotationMatrix = float3x3(
        cosAngle, -sinAngle, 0.0,   // First row
        sinAngle,  cosAngle, 0.0,   // Second row
        0.0,       0.0,      1.0    // Third row
    );
    
    float3 rotatedVertex = mul(rotationMatrix, vertex);
    
    vertices[index].position = rotatedVertex;
}
