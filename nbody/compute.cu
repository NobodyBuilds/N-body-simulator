


#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "compute.h"

#include <atomic>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <iostream>

#include <math_constants.h>
#include <math_functions.h>
#include "settings.h"

#include <cub/cub.cuh>

#define BLOCKS(n) ((n + 255) / 256)
#define THREADS 256
#define MAX_PARTICLES_PER_CELL 256
// param settings;

static void* d_sortTempStorage = nullptr;
static size_t sortTempBytes = 0;
static int* d_particleHash_alt = nullptr; // double buffer
static int* d_particleIndex_alt = nullptr;
int HASH_TABLE_SIZE; // 2^18 - adjust based on particle count
int d_count;

// debug
__device__ int min_nb, max_nb, avg_nb = 0;
__device__ float min_Density, max_Density, avg_Density = 0;
__device__ float min_nearDensity, max_nearDensity, avg_nearDensity = 0;

// device helpers
__host__ __device__ inline float clamp(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}
__host__ __device__ inline float3 operator+(float3 a, float3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
__host__ __device__ inline float3 operator-(float3 a, float3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
__host__ __device__ inline float3 operator*(float3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
__host__ __device__ inline float3 operator/(float3 a, float s) { return { a.x / s, a.y / s, a.z / s }; }
__host__ __device__ inline float3 operator*(float s, float3 a)
{
    return { a.x * s, a.y * s, a.z * s };
}
__host__ __device__ inline float3 operator-(float3 v)
{
    return { -v.x, -v.y, -v.z };
}
__host__ __device__ inline float3& operator+=(float3& a, const float3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
__host__ __device__ inline float4& operator+=(float4& a, const float4& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

__host__ __device__ inline float3& operator-=(float3& a, const float3& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}
__host__ __device__ inline float3& operator*=(float3& v, float s)
{
    v.x *= s;
    v.y *= s;
    v.z *= s;
    return v;
}
__host__ __device__ inline float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float length(float3 v)
{
    return sqrt(dot(v, v));
}
__host__ __device__ inline float3 normalize(float3 v)
{
    float l = length(v);
    return (l > 0.0f) ? v / l : float3{ 0, 0, 0 };
}
__device__ __forceinline__ float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

// arrays to store particle data

// float4 for better cache
// compact array system with 124bit load 4X better cache loading
// uses 68bytes per particle ,before was 80bytes

float4* positions = nullptr; // contains- x,y,x,density

float4* velocity = nullptr; // vx,vy,vz,free

float4* accelration = nullptr; // ax,ay,az,heat in heat

float4* pdata = nullptr; //size,mass,h,idk

int* isstar = nullptr; // 1 if star, 0 if not


// stroage for sorting
float4* positions_sorted = nullptr;
float4* velocity_sorted = nullptr;
float4* accelration_sorted = nullptr;
float4* pdata_sorted = nullptr;


// constant struct for kernels to reduce memory occupancy and register load, also to avoid passing too many parameters to kernels which can cause register spilling and performance 

__constant__ data d;
data h;



extern "C" void syncstruct() {
    h.dt = settings.fixedDt;
    h.particlemass = settings.particleMass;
    h.h = settings.h;
    h.sdensity = settings.Sdensity;
    h.h2 = settings.h2;
    h.pollycoef6 = settings.pollycoef6;
    h.spikycoef = settings.spikycoef;
    h.pressure = settings.pressure;
    h.restDensity = settings.rest_density;
    h.spikyGradv = settings.spikygradv;
    h.viscK = settings.visc;
    h.viscstrength = settings.viscosity;
    h.viscK = settings.visc;
    h.count = settings.count;
    h.G = settings.G;
    h.centermass = settings.centermass;
    h.orbitalspeed = settings.orbitspeed;
    h.radius = settings.radius;
    h.maxradius = settings.maxradius;
    h.starsize = settings.starsize;
    h.mode = settings.mode;
    h.screenHeight = settings.screenHeight;
    h.screenHeight = settings.screenHeight;
    h.dst = settings.dst;
    h.impactspeed = settings.impactspeed;
    h.yspeed = settings.yspeed;
    h.lockstar = settings.lockstar;
    cudaMemcpyToSymbol(d, &h, sizeof(data));

}

extern "C" bool initgpu(int count)
{

    cudaMalloc(&positions, count * sizeof(float4));
    cudaMalloc(&velocity, count * sizeof(float4));
    cudaMalloc(&accelration, count * sizeof(float4));

    cudaMalloc(&positions_sorted, count * sizeof(float4));
    cudaMalloc(&velocity_sorted, count * sizeof(float4));
    cudaMalloc(&accelration_sorted, count * sizeof(float4));
    cudaMalloc(&isstar, count * sizeof(int));
    cudaMalloc(&pdata, count * sizeof(float4));
	cudaMalloc(&pdata_sorted, count * sizeof(float4));



    printf("Total particle mem allocated: %.2f MB\n", (count * (8 * sizeof(float4) + (1 * sizeof(int)))) / (1024.0 * 1024.0)); // prints the mem size for total allocation with maxpartiucles buffer

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("ERROR: CUDA mem allocation failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}
extern "C" void freegpu()
{
    cudaFree(positions);
    cudaFree(positions_sorted);
    cudaFree(velocity);
    cudaFree(velocity_sorted);
    cudaFree(accelration);
    cudaFree(accelration_sorted);
    cudaFree(isstar);
    cudaFree(pdata);


    pdata = nullptr;
    isstar = nullptr;
    positions = nullptr;
    velocity = nullptr;
    accelration = nullptr;
    accelration_sorted = nullptr;
};

struct GLVertex
{
    float px, py, pz;
    float radius;
    float cr, cg, cb, ca;
    float ox, oy;
    float wx, xy, xz;
};

static cudaGraphicsResource* g_vboResource = nullptr;

extern "C" void registerGLBuffer(unsigned int vboId)
{
    cudaError_t err = cudaGraphicsGLRegisterBuffer(
        &g_vboResource, vboId, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess)
        printf("ERROR: cudaGraphicsGLRegisterBuffer: %s\n", cudaGetErrorString(err));
    else
        printf("INFO: GL VBO %u registered with CUDA\n", vboId);
}

extern "C" void unregisterGLBuffer()
{
    if (g_vboResource)
    {
        cudaGraphicsUnregisterResource(g_vboResource);
        g_vboResource = nullptr;
    }
}

__global__ void packToVBOKernel(
    int n,
    const float4* __restrict__ pos,
    const float4* __restrict__ vel,
    float4* acl,

    GLVertex* vbo, bool heat, float heatMultipler, float dt, float heatDecay, float4* size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float4 p = __ldg(&pos[i]);
    float4 v = __ldg(&vel[i]);
    float4 a = acl[i];

    int3 c = { 0, 0, 0 };

    if (heat)
    {

        float speed = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

        float heat = (speed + size[i].y) * 0.5f;
        // heat += mass[i];
        a.w += (heat * heatMultipler) * dt; // acl.w=heat;

        a.w *= expf(-heatDecay * dt);
        a.w = clamp(a.w, 0.0f, 100.0f);
        // load once to get col.w
        float t = clamp(a.w / 100.0f, 0.0f, 1.0f);
        t = pow(t, 0.95f);
        acl[i].w = a.w;

        if (t < 0.35f)
        {
            float tt = t / 0.25f;
            c.x = 0.0f;
            c.y = tt * 255.0f;
            c.z = 255.0f;
        }
        else if (t < 0.55f)
        {
            float tt = (t - 0.25f) / 0.25f;
            c.x = 0.0f;
            c.y = 255.0f;
            c.z = (1.0f - tt) * 255.0f;
        }
        else if (t < 0.85f)
        {
            float tt = (t - 0.5f) / 0.25f;
            c.x = tt * 255.0f;
            c.y = 255.0f;
            c.z = 0.0f;
        }
        else
        {
            float tt = (t - 0.75f) / 0.25f;
            c.x = 255.0f;
            c.y = (1.0f - tt) * 255.0f;
            c.z = 0.0f;
        }
    }

    float fpx = p.x,
        fpy = p.y,
        fpz = p.z;
    float rad = __ldg(&size[i].x); // size
    float fcr = c.x * (1.0f / 255.0f);
    float fcg = c.y * (1.0f / 255.0f);
    float fcb = c.z * (1.0f / 255.0f);

    // Matches the offsets used in the old CPU drawAll() loop
    const float ox[3] = { -1.0f, 3.0f, -1.0f };
    const float oy[3] = { -1.0f, -1.0f, 3.0f };

    int base = i * 3;
    for (int k = 0; k < 3; k++)
    {
        GLVertex& vtx = vbo[base + k];
        vtx.px = fpx;
        vtx.py = fpy;
        vtx.pz = fpz;
        vtx.radius = rad;
        vtx.cr = fcr;
        vtx.cg = fcg;
        vtx.cb = fcb;
        vtx.ca = 1.0f;
        vtx.ox = ox[k];
        vtx.oy = oy[k];
        vtx.wx = 0.0f;
        vtx.xy = 0.0f;
        vtx.xz = 0.0f;
    }
}

/// ///////////////////////////
// sph

struct HashCell
{
    int count;                             // Number of particles in this cell
    int particles[MAX_PARTICLES_PER_CELL]; // Particle indices
};
static int* d_cellStart = nullptr;
static int* d_cellEnd = nullptr;
static int* d_particleHash = nullptr;
static int* d_particleIndex = nullptr;

__device__ __host__ inline unsigned int spatialHash(int ix, int iy, int iz, int HASH_TABLE_SIZE)
{

    const unsigned int p1 = 73856093;
    const unsigned int p2 = 19349663;
    const unsigned int p3 = 83492791;

    unsigned int hash = ((unsigned int)ix * p1) ^
        ((unsigned int)iy * p2) ^
        ((unsigned int)iz * p3);

    return hash & (HASH_TABLE_SIZE - 1);
}

__device__ __host__ inline void getCell(float x, float y, float z,
    float cellSize,
    int& ix, int& iy, int& iz)
{
    ix = (int)floorf(x / cellSize);
    iy = (int)floorf(y / cellSize);
    iz = (int)floorf(z / cellSize);
}

__device__ __host__ inline unsigned int getHashFromPos(float x, float y, float z,
    float cellSize, int hs)
{
    int ix, iy, iz;
    getCell(x, y, z, cellSize, ix, iy, iz);
    return spatialHash(ix, iy, iz, hs);
}

extern "C" bool initDynamicGrid(int maxParticles)
{
    // using maxpartcles which are 2-5X total particles for emiiter to work and dynamic add or remove particles
    HASH_TABLE_SIZE = 1;
    while (HASH_TABLE_SIZE < maxParticles * 2)
        HASH_TABLE_SIZE <<= 1;
    //  size_t hashTableBytes = HASH_TABLE_SIZE * sizeof(HashCell);

    // cudaMalloc(&d_hashTable, hashTableBytes);
    cudaMalloc(&d_cellStart, HASH_TABLE_SIZE * sizeof(int));
    cudaMalloc(&d_cellEnd, HASH_TABLE_SIZE * sizeof(int));
    cudaMalloc(&d_particleHash, maxParticles * sizeof(int));
    cudaMalloc(&d_particleIndex, maxParticles * sizeof(int));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("ERROR: CUDA allocation failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    // Initialize
    //  cudaMemset(d_hashTable, 0, hashTableBytes);
    cudaMemset(d_cellStart, -1, HASH_TABLE_SIZE * sizeof(int));
    cudaMemset(d_cellEnd, -1, HASH_TABLE_SIZE * sizeof(int));

    cudaMalloc(&d_particleHash_alt, maxParticles * sizeof(int));
    cudaMalloc(&d_particleIndex_alt, maxParticles * sizeof(int));

    // correct dry run — all nullptr, just getting the size
    cub::DeviceRadixSort::SortPairs(
        nullptr, sortTempBytes,
        (int*)nullptr, (int*)nullptr,
        (int*)nullptr, (int*)nullptr,
        maxParticles);

    cudaMalloc(&d_sortTempStorage, sortTempBytes);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("ERROR: CUDA sort temp allocation failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}
extern "C" void freeDynamicGrid()
{
    // cudaFree(d_hashTable);
    cudaFree(d_cellStart);
    cudaFree(d_cellEnd);
    cudaFree(d_particleHash);
    cudaFree(d_particleIndex);

    cudaFree(d_particleHash_alt);
    cudaFree(d_particleIndex_alt);
    cudaFree(d_sortTempStorage);
    d_particleHash_alt = nullptr;
    d_particleIndex_alt = nullptr;
    d_sortTempStorage = nullptr;
    sortTempBytes = 0;

    d_cellStart = nullptr;
    d_cellEnd = nullptr;
    d_particleHash = nullptr;
    d_particleIndex = nullptr;
}

__global__ void computeHashKernel(
    int numParticles,
    float cellSize,
    const float4* __restrict__ pos,
    int* particleHash,
    int* particleIndex,
    int hs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles)
        return;

    float x = pos[i].x;
    float y = pos[i].y;
    float z = pos[i].z;

    // Check for NaN
    if (isnan(x) || isnan(y) || isnan(z))
    {
        particleHash[i] = 0xFFFFFFFF; // Invalid hash
        particleIndex[i] = i;
        printf("WARNING nan positions\n");
        return;
    }

    // Compute hash
    unsigned int hash = getHashFromPos(x, y, z, cellSize, hs);

    particleHash[i] = hash;
    particleIndex[i] = i; // Store original index
}

__global__ void findCellBoundariesKernel(
    int numParticles,
    int* particleHash,
    int* cellStart,
    int* cellEnd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles)
        return;

    unsigned int hash = particleHash[i];

    unsigned int prevHash = (i > 0) ? particleHash[i - 1] : 0xFFFFFFFF;

    if (hash != prevHash)
    {
        // Start of new cell
        cellStart[hash] = i;
        if (i > 0)
        {
            cellEnd[prevHash] = i;
        }
    }

    if (i == numParticles - 1)
    {
        cellEnd[hash] = numParticles;
    }
}
__global__ void reorderParticlesKernel(
    int n, float dt,
    const int* __restrict__ sortedIndex, // d_particleIndex (after CUB sort)
    const float4* __restrict__ posIn,
    const float4* __restrict__ velIn,
    const float4* __restrict__ aclIn,
    const float4* __restrict__ pd,

    float4* posOut,
    float4* velOut,
    float4* aclOut,
	float4* pdOut

)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    int src = sortedIndex[i]; // where this particle CAME from in the original array
    float4 pi = __ldg(&posIn[src]);
    float4 vi = __ldg(&velIn[src]);
    float4 ai = __ldg(&aclIn[src]);
	float4 pdi = __ldg(&pd[src]);
    // using pridicted positiopn into sorted arrays for density and pressure kernel  and help in stability
    // directly writeing to sorted arrays to avoid extra copy and also we will be using predicted position for density and pressure calculation which will help in stability
    float px = pi.x + vi.x * dt;
    float py = pi.y + vi.y * dt;
    float pz = pi.z + vi.z * dt;

    posOut[i] = { px, py, pz, pi.w };
    velOut[i] = vi;
    aclOut[i] = ai;
	pdOut[i] = pdi;
}

__global__ void clearActiveCellsKernel(
    int numParticles,
    const int* __restrict__ particleHash, // sorted hashes from LAST frame
    int* cellStart,
    int* cellEnd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles)
        return;

    // Each thread clears its own hash bucket.
    // Duplicate writes (multiple particles same cell) are harmless — idempotent.
    unsigned int h = (unsigned int)particleHash[i];
    cellStart[h] = -1;
    cellEnd[h] = -1;
}

void buildDynamicGrid(

    float cellSize,
    const float4* __restrict__ pos, float dt

)
{
    int blocks = (settings.count + THREADS - 1) / THREADS;
    int numParticles = settings.count;
    if (numParticles <= 0)
    {
        printf("WARNING: buildDynamicGrid called with %d particles\n", numParticles);
        return;
    }

    static bool firstFrame = true;
    if (firstFrame)
    {
        cudaMemset(d_cellStart, -1, HASH_TABLE_SIZE * sizeof(int));
        cudaMemset(d_cellEnd, -1, HASH_TABLE_SIZE * sizeof(int));
        firstFrame = false;
    }
    else
    {
        // d_particleHash still holds last frame's sorted hashes — perfect
        clearActiveCellsKernel << <blocks, THREADS >> > (
            numParticles, d_particleHash, d_cellStart, d_cellEnd);
    }
    computeHashKernel << <blocks, THREADS >> > (numParticles, cellSize, pos, d_particleHash, d_particleIndex, HASH_TABLE_SIZE);

    // sorting

    cub::DeviceRadixSort::SortPairs(
        d_sortTempStorage, sortTempBytes,
        d_particleHash, d_particleHash_alt,
        d_particleIndex, d_particleIndex_alt,
        settings.count);
    std::swap(d_particleHash, d_particleHash_alt);
    std::swap(d_particleIndex, d_particleIndex_alt);

    if (d_particleHash == nullptr || d_particleIndex == nullptr)
    {
        printf("ERROR: Null pointers in grid sort!\n");
        return;
    }

    findCellBoundariesKernel << <blocks, THREADS >> > (numParticles, d_particleHash, d_cellStart, d_cellEnd);
    // sorteding arrays
    reorderParticlesKernel << <blocks, THREADS >> > (
        numParticles, dt,
        d_particleIndex,     // tells us: sorted slot i came from original slot src
        positions, velocity, accelration,pdata, // source
        positions_sorted, velocity_sorted, accelration_sorted,pdata_sorted);
}

// sph-functions

__global__ void computeDensity(float cellSize, float4* pos, int hs, const int* __restrict__ cellstart, const int* __restrict__ cellend, const int* __restrict__ particleindex, float4* pdata)
{
    // no shared memory because the arrays are sorted and coalesced access is good enough, also we are doing more computation per neighbor which helps hide latency.
    // shared memory has been tried and got no difference
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count)
        return;

    float4 p = __ldg(&pos[i]); // is it good using ldg?? idk
    float xi = p.x;
    float yi = p.y;
    float zi = p.z;

    //  float rho = density[i];
    // float rhon = neardensity[i];
    int cx, cy, cz;
    getCell(xi, yi, zi, cellSize, cx, cy, cz);



    float m_i = __ldg(&pdata[i].y);

    float rho = m_i * d.sdensity;

    // Search 27 neighboring cells
    for (int dz = -1; dz <= 1; dz++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                /*int manhattanDist = abs(dx) + abs(dy) + abs(dz);
                if (manhattanDist > 2) continue;*/
                unsigned int hash = spatialHash(cx + dx, cy + dy, cz + dz, hs);

                int start = cellstart[hash];
                int end = cellend[hash];
                if (start == -1)
                    continue;


                if (start > 0)

                    /*  if (debug && count > 0) {
                         printf("  Neighbor cell (%d,%d,%d) hash=%u: %d particles\n",
                              cx + dx, cy + dy, cz + dz, hash, count);
                      }*/

                    for (int k = start; k < end; k++)
                    {
                        int j = k; // particleindex[k]; // index of neighbor particle in sorted array

                        if (j == i)
                            continue;
                        float4 pj = __ldg(&pos[j]);
                        float dx_val = xi - pj.x;
                        float dy_val = yi - pj.y;
                        float dz_val = zi - pj.z;
                        float r2 = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;

                        if (r2 < d.h2)
                        {
                            float invR = rsqrtf(r2 + 1e-12f);
                            float r = r2 * invR;
                            float v = d.h2 - r2;
                            // float v2 = h - r;
                            float vcube = v * v * v;

                            float D = d.pollycoef6 * vcube; // precomputed pollycoef6
                            // float d = spikycoef2 * v2 * v2;
                            float m_j = __ldg(&pdata[j].y); // mass
                            rho += m_j * D;
                        }
                    }
            }
        }
    }
    /* if (debug) {
         printf("density:%5f \n",
              rho);
     }*/

    pos[i].w = fmaxf(rho, 1e-6f);
    //pos[i].w = rho;
    //vel[i].w = rhon;
}

__global__ void computePressure(float cellSize, const float4* __restrict__ pos, float4* acl, float4* vel, int hs, int* cellstart, int* cellend, int* particleIndex, float4* pdata)
{

    // no shared memory because the arrays are sorted and coalesced access is good enough, also we are doing more computation per neighbor which helps hide latency.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count)
        return;

    float4 p = __ldg(&pos[i]);
    float4 v = __ldg(&vel[i]);
    float xi = p.x;
    float yi = p.y;
    float zi = p.z;

    float3 force = { 0.0f, 0.0f, 0.0f };


    float p_i = d.pressure * (p.w - d.restDensity);

    float3 visc = { 0.0f, 0.0f, 0.0f };

    float3 vi = make_float3(v.x, v.y, v.z);
    int cx, cy, cz;
    getCell(xi, yi, zi, cellSize, cx, cy, cz);


    float rho_i = p.w;


    float pressuretermRho_i = p_i / (rho_i * rho_i);

    for (int dz = -1; dz <= 1; dz++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                /* int manhattanDist = abs(dx) + abs(dy) + abs(dz);
                 if (manhattanDist > 2) continue;*/
                 // TRY 3 INSTEAD OF 2 TO FIX JITTERYNESS

                unsigned int hash = spatialHash(cx + dx, cy + dy, cz + dz, hs);

                int start = cellstart[hash];
                int end = cellend[hash];
                if (start == -1)
                    continue;


                if (start > 0)

                    for (int k = start; k < end; k++)
                    {
                        int j = k; // particleIndex[k]; // index of neighbor particle in sorted array

                        if (j == i)
                            continue;
                        float4 pj = __ldg(&pos[j]);
                        float4 vj = __ldg(&vel[j]);
                        float dx_val = xi - pj.x;
                        float dy_val = yi - pj.y;
                        float dz_val = zi - pj.z;
                        float r2 = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;

                        if (r2 < d.h2 && r2 > 1e-9f)
                        {

                            float invR = rsqrtf(r2 + 1e-12f);
                            float r = r2 * invR;

                            float p_j = d.pressure * (pj.w - d.restDensity);

                            float3 dir = { dx_val * invR, dy_val * invR, dz_val * invR };


                            /*  if (debug && neighborCount <= 3) {
                                  printf("  Neighbor %d: dist=%.3f rho=%.6f p=%.6f\n",
                                      j, r, density[j], p_j);
                              }*/
                            float rho_j = pj.w;
                            float x = d.h - r;
                            float gradW = d.spikyGradv * x * x; // precomputed gradw in negative value

                            float pressureterm = pressuretermRho_i + p_j / (rho_j * rho_j);
                            // float pressureterm = (p_i + p_j)/2;

                            float m_j = __ldg(&pdata[j].y); // particle mass

                            force += -m_j * pressureterm * gradW * dir;

                            float4 v2 = __ldg(&vel[j]);
                            float3 vj = make_float3(v2.x, v2.y, v2.z);
                            float3 vij = (vj - vi);

                            float lapW = d.viscK * x;
                            float viscosityCoeff = d.viscstrength;
                            visc += viscosityCoeff * m_j * vij / rho_j * lapW;




                        }
                    }
            }
        }
    }

    float4 accl;
    float4 delta;

    accl.z = (force.z + visc.z) / rho_i;
    accl.x = (force.x + visc.x) / rho_i;
    accl.y = (force.y + visc.y) / rho_i;
    accl.w = 0.0f;



    // int org = particleIndex[i]; // where this particle came from in the original unsorted array
                                 // velocity written to org idx ,using swaps or memcpy caused visuals errors and performance heavy
    acl[i] += accl; // write back to original slot in acl array
    // velocity verlet intigration fisrt step
    /*velocity[org].x += accl.x * dt * 0.5;
    velocity[org].y += accl.y * dt * 0.5;
    velocity[org].z += accl.z * dt * 0.5;*/
}


__global__ void scatterarray(int numParticles, float dt, float4* aclin, float4* aclout, float4* vel, int* particleindex) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles)
        return;
    int org = particleindex[i]; // where this particle came from in the original unsorted array
    float4 accl = __ldg(&aclin[i]);
    aclout[org] = accl;
    float4 v;
    v.x = accl.x * dt * 0.5;
    v.y = accl.y * dt * 0.5;
    v.z = accl.z * dt * 0.5;

    vel[org] += v;


}

__device__ void atomicMinFloat(float* addr, float val)
{
    int* addr_i = (int*)addr;
    int old = *addr_i, assumed;
    do
    {
        assumed = old;
        if (__int_as_float(assumed) <= val)
            break;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
}

__device__ void atomicMaxFloat(float* addr, float val)
{
    int* addr_i = (int*)addr;
    int old = *addr_i, assumed;
    do
    {
        assumed = old;
        if (__int_as_float(assumed) >= val)
            break;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
}

__global__ void debug(int n, float4* pos, float4* vel, int* ncount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    int nc = ncount[i];
    float d = pos[i].w;
    float nd = vel[i].w;

    atomicMin(&min_nb, nc);
    atomicMax(&max_nb, nc);
    atomicAdd(&avg_nb, nc);

    atomicMinFloat(&min_Density, d);
    atomicMaxFloat(&max_Density, d);
    atomicAdd(&avg_Density, d);

    atomicMinFloat(&min_nearDensity, nd);
    atomicMaxFloat(&max_nearDensity, nd);
    atomicAdd(&avg_nearDensity, nd);
}
//gravity

__global__ void gravityKernel(float4* pos, float4* acl, float4* pdata) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count)
        return;

    float4 p = __ldg(&pos[i]);
    float m_i = __ldg(&pdata[i].y);
    float4 a = { 0.0f,0.0f,0.0f,0.0f };
    for (int j = 0; j < d.count; j++) {
        if (j == i)continue;
        float4 pj = __ldg(&pos[j]);
        float m_j = __ldg(&pdata[j].y);
        float dx = pj.x - p.x;
        float dy = pj.y - p.y;
        float dz = pj.z - p.z;
        float dist = dx * dx + dy * dy + dz * dz + (0.1f * d.h) * (0.1f * d.h);
        float invdist = rsqrtf(dist);
        float inv3 = invdist * invdist * invdist;

        float force = d.G * (m_j * inv3);
        a.x += (force*dx) /m_i;
        a.y += (force*dy) /m_i;
        a.z += (force*dz) /m_i;

    }
    a.w = 0.0f;
    acl[i] += a;


}

__global__ void starCollisionKernel(float4* pos, float4* vel, float4* pdata, int* star) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d.count) return;
    if (star[j] == 1) return;
    float4 pj = __ldg(&pos[j]);
    float4 vj = __ldg(&vel[j]);
    for (int i = 0; i < d.count; i++) {
        if (star[i] == 0) continue;
        float4 pi = __ldg(&pos[i]);
        float3 r = make_float3(pj.x - pi.x, pj.y - pi.y, pj.z - pi.z);
        float dist = length(r);
        float minDist = pdata[i].x + pdata[j].x;

        if (dist < minDist) {
            float3 n = r / (dist + 1e-6f);
            pos[j] = make_float4(pi.x + n.x * minDist, pi.y + n.y * minDist, pi.z + n.z * minDist, pj.w);
            float vn = dot(make_float3(vj.x, vj.y, vj.z), n);
            if (vn < 0.0f)
                vel[j] = make_float4(vj.x - n.x * vn, vj.y - n.y * vn, vj.z - n.z * vn, vj.w);
        }
    }
}

//intigration and update
__global__ void updateKernel(float dt, float4* pos, float4* vel, float4* acl, int* star
)
{
    // Vec3 acc_new;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count)
        return;

    float4 p = __ldg(&pos[i]);
    float4 vl = __ldg(&vel[i]);
    float4 a = __ldg(&acl[i]);
    vl.x += a.x * dt * 0.5f;
    vl.y += a.y * dt * 0.5f;
    vl.z += a.z * dt * 0.5f;



    /*if (ncount[i] < 5)
    {
        float drag = expf(-coeff * dt);
        vl.x *= drag;
        vl.y *= drag;
        vl.z *= drag;
    }*/

    if (d.lockstar && star[i] == 1) {
        vl.x = 0.0f;
        vl.y = 0.0f;
        vl.z = 0.0f;
    }
    p.x += vl.x * dt;
    p.y += vl.y * dt;
    p.z += vl.z * dt;

    a.x = 0;
    a.y = 0;
    a.z = 0;


    pos[i] = p;
    vel[i] = vl;
    acl[i] = a;
}
//spawning
__device__ inline float rand01(unsigned int& seed)
{
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return (float)(seed & 0xFFFFFF) / (float)0xFFFFFF;
}
__device__ float randf(unsigned int& state, float min, float max) {
    return min + (max - min) * rand01(state);
}
__global__ void registerKernel(float4* position, float4* velocity, float4* accelration, int* isstar, float4* pdata)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count) return;







    unsigned int key = i * 1234567u;
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float vx = 0.0f, vy = 0.0f, vz = 0.0f;
    float mass = 1.0f, size = 1.0f;
    int   iscenter = 0;

    // ===============================
    // MODE 0 — single star disc 
    // ===============================
    if (d.mode == 0)
    {
        if (i == 0)
        {
            mass = d.centermass;
            size = d.starsize;
            iscenter = 1;
            // x y z stay 0 — star at world origin
        }
        else
        {
            float angle = randf(key, 0.0f, 2.0f * CUDART_PI_F);
            float r = randf(key, d.radius, d.maxradius);


            x = cosf(angle) * r;
            z = sinf(angle) * r;
            y = randf(key, -2.5f, 2.5f);


            float v = sqrtf(d.G * d.centermass / r) * d.orbitalspeed;
            vx = sinf(angle) * v;
            vz = -cosf(angle) * v;
            vy = randf(key, -0.05f, 0.05f) * v;
        }
    }

    // ===============================
    // MODE 1 — double star system 
    // ===============================
    if (d.mode == 1)
    {
        int half = d.count / 2;


        float halfSep = d.maxradius * 2.0f;
        float3 c1 = { -halfSep, 0.0f, 0.0f };
        float3 c2 = { halfSep, 0.0f, 0.0f };

        if (i == 0 || i == half)
        {
            mass = d.centermass;
            size = d.starsize;
            iscenter = 1;
            float3 c = (i == 0) ? c1 : c2;
            x = c.x; y = 0.0f; z = c.z;
        }
        else
        {
            float3 c = (i < half) ? c1 : c2;
            float angle = randf(key, 0.0f, 2.0f * CUDART_PI_F);
            float r = randf(key, d.radius, d.maxradius);

            x = c.x + cosf(angle) * r;
            z = c.z + sinf(angle) * r;
            y = randf(key, -3.0f, 3.0f);

            float v = sqrtf(d.G * d.centermass / r) * d.orbitalspeed;
            vx = -sinf(angle) * v;
            vz = cosf(angle) * v;
        }
    }

    // ===============================
    // MODE 2 — star + spherical cloud
    // ===============================
    if (d.mode == 2)
    {
        if (i == 0)
        {
            mass = d.centermass;
            size = d.starsize;
            iscenter = 1;
            // x y z stay 0
        }
        else
        {
            float u = rand01(key);
            float v = rand01(key);
            float w = rand01(key);
            float theta = u * 2.0f * CUDART_PI_F;
            float phi = acosf(2.0f * v - 1.0f);
            float rad = cbrtf(w) * d.radius * 5.0f;

            x = rad * sinf(phi) * cosf(theta);
            z = rad * sinf(phi) * sinf(theta);
            y = rad * cosf(phi);

            vx = randf(key, -0.1f, 0.1f);
            vz = randf(key, -0.1f, 0.1f);
            vy = randf(key, -0.1f, 0.1f);
        }
    }

    // ===============================
    // MODE 4 — earth / theia impact
    // ===============================
    if (d.mode == 3)
    {
        int   earthCount = (int)(d.count * 0.85f);
        int   theiaCount = d.count - earthCount;
        float spacing = d.h * 0.8f;
        int   earth_side = (int)ceilf(cbrtf((float)earthCount));

        if (i < earthCount)
        {
            int ix = i % earth_side;
            int iy = (i / earth_side) % earth_side;
            int iz = i / (earth_side * earth_side);
            x = ix * spacing;
            z = iy * spacing;
            y = iz * spacing;
            // vx vy vz stay 0
        }
        else
        {
            int theia_side = (int)ceilf(cbrtf((float)theiaCount));
            int ti = i - earthCount;
            int ix = ti % theia_side;
            int iy = (ti / theia_side) % theia_side;
            int iz = ti / (theia_side * theia_side);

            float earth_width = earth_side * spacing;
            float separation = earth_width + d.dst;

            x = separation + ix * spacing;
            z = iy * spacing;
            y = iz * spacing;
            vx = -d.impactspeed * cosf(CUDART_PI_F / 4.0f);
            vz = d.yspeed * sinf(CUDART_PI_F / 4.0f);
        }
    }


    position[i] = { x,    y,    z,    0.0f };
    velocity[i] = { vx,   vy,   vz,   0.0f };
    accelration[i] = { 0.0f, 0.0f, 0.0f, 0.0f };
    isstar[i] = iscenter;
    pdata[i] = { size, mass, 0.0f, 0.0f };
}


extern "C" void registerBodies()
{
    int Block = (settings.count + THREADS - 1) / THREADS;
    registerKernel << <Block, THREADS >> > (
        positions, velocity, accelration, isstar, pdata);
}

extern "C" void computephysics(float dt)
{
    int blocks = (settings.count + THREADS - 1) / THREADS;
    int totalBodies = settings.count;
    // cudaError_t err;
    float d_cellsize = settings.h * settings.cellSize; // tweaak it gng

    //  float subdt = settings.fixedDt / settings.substeps;
    float deltaTime = dt / settings.substeps;


    if (settings.nopause)
    {
        for (int i = 0; i < settings.substeps; i++)
        {
            // update positipons

            updateKernel << <blocks, THREADS >> > (deltaTime, positions, velocity, accelration, isstar);
            // acelrations reset

            if (settings.sph || settings.gravity) {
                buildDynamicGrid(d_cellsize, positions, deltaTime);
            }
            if (settings.sph)
            {
                // builds grid and sorted arrays with pridicted positions

                // uses p pos for stability
                computeDensity << <blocks, THREADS >> > (d_cellsize, positions_sorted, HASH_TABLE_SIZE, d_cellStart, d_cellEnd, d_particleIndex, pdata_sorted);

                // reads from pridicted pos and writes back to orginal velocity array with velocity verlet 2nd step
                computePressure << <blocks, THREADS >> > (d_cellsize, positions_sorted, accelration_sorted, velocity_sorted, HASH_TABLE_SIZE, d_cellStart, d_cellEnd, d_particleIndex, pdata_sorted);


            }
            if (settings.gravity) {
                gravityKernel << <blocks, THREADS >> > (positions_sorted, accelration_sorted, pdata_sorted);
            }

            scatterarray << <blocks, THREADS >> > (totalBodies, deltaTime, accelration_sorted, accelration, velocity, d_particleIndex);

            starCollisionKernel << <blocks, THREADS >> > (positions, velocity, pdata, isstar);
        }

    }


    //// cudaEventRecord(start);
    //if (g_vboResource)
    //{
    //    cudaGraphicsMapResources(1, &g_vboResource, 0);

    //    GLVertex* d_vbo = nullptr;
    //    size_t nbytes = 0;
    //    cudaGraphicsResourceGetMappedPointer((void**)&d_vbo, &nbytes, g_vboResource);

    //    packToVBOKernel << <blocks, THREADS >> > (
    //        settings.count, positions, velocity, accelration, d_vbo, settings.heateffect, settings.heatMultiplier, deltaTime, settings.cold, pdata);
    //}
}

extern "C" void render() {

    int blocks = (settings.count + THREADS - 1) / THREADS;

    cudaGraphicsMapResources(1, &g_vboResource, 0);

    GLVertex* d_vbo = nullptr;
    size_t nbytes = 0;
    cudaGraphicsResourceGetMappedPointer((void**)&d_vbo, &nbytes, g_vboResource);

    packToVBOKernel << <blocks, THREADS >> > (
        settings.count, positions, velocity, accelration,
        d_vbo, settings.heateffect, settings.heatMultiplier,
        settings.fixedDt, settings.cold, pdata);


    cudaGraphicsUnmapResources(1, &g_vboResource, 0);
}

__global__ void changestarsizeKernel(float4* pdata, int* isstar, float newsize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count)
        return;
    if (isstar[i]) // identify star by mass
    {
        pdata[i].x = newsize; // update size in pdata
    }
}

extern "C" void changestarsize() {

    int blocks = (settings.count + THREADS - 1) / THREADS;
    changestarsizeKernel << <blocks, THREADS >> > (pdata, isstar, settings.starsize);
}

__global__ void changestarmassKernel(float4* pdata, int* isstar, float newmass) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count)
        return;
    if (isstar[i]) // identify star by mass
    {
        pdata[i].y = newmass; // update mass in pdata
    }
}

extern "C" void changestarmass() {
    int blocks = (settings.count + THREADS - 1) / THREADS;
    changestarmassKernel << <blocks, THREADS >> > (pdata, isstar, settings.centermass);
}