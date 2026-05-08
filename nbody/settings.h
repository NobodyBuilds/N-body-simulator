#pragma once
struct param {
	// === FLOAT VARIABLES (4 bytes each) ===
	// Grouped for cache locality and alignment
	float fixedDt = 1 / 60.0f;
	float simspeed = 1.0f;
	float size = 1.0f;
	float particleMass = 0.20f;
	float cold = 4.500f;
	float heatMultiplier = 15.0f;
	float h = 4.0f;
	float h2 = h * h;
	float rest_density = 0.01800f;
	float pressure = 200.0f;
	float nearpressure = 450.0f;
	float visc = 0.0529f;
	float gravityforce = 60.0f;
	float pi = 3.14159265358979323846f;
	float pollycoef6 = 0.0f;
	float spikycoef = 0.0f;
	float Sdensity = 0.0f;
	float cellSize = 1.0f;
	float nearRestDensity = 0.153f;//not used atall wasted variable :D
	float ndensity = 0.0f;
	float spikygradv = 0.0f;
	float viscosity = 0.0f;
	float wx = 0.0f;
	float wy = 0.0f;
	float wz = 0.0f;
	float avgFps = 0.0f;
	float minFps = 1000.0f;
	float maxFps = 0.0f;
	float fpsTimer = 0.0f;
	float accumulator = 0.0f;
	float fps = 0.0f;
	float centermass = 100.0f;
	float starsize = 3.0f;
	float radius = 100.0f;
	float maxradius = 300.0f;
	float orbitspeed = 0.5f;
	float G = 6.67f;

	float min_density, max_density, avg_density = 0;
	float min_neardensity, max_neardensity, avg_neardensity = 0;


	double fuc_ms = 0.0;
	// === INT VARIABLES (4 bytes each) ===
	int totalBodies = 60000;
	int count = totalBodies;
	int min_n, max_n, avg_n = 0;
	int substeps = 2;
	int fpsCount = 0;
	int mode = 0;
	int screenWidth;
	int screenHeight;

	// === BOOL VARIABLES (1 byte each) ===
	bool sph = true;
	bool nopause = true;
	bool heateffect = true;
	bool debug = false;
	bool recordSim = false;
	bool gui = true;
};
extern param settings;

struct data {
	float dt;
	float particlemass;
	float h;
	float spacing;
	float ndensity;
	float sdensity;
	float h2;
	float pollycoef6;
	float spikycoef;
	float pressure;
	float nearpressure;
	float restDensity;
	float nearrestdensity;
	float spikyGradv;
	float viscK;
	float viscstrength;
	float G;
	float orbitalspeed;
	float radius;
	float centermass;
	float starsize;
	float maxradius;
	int mode;
	int count;
	int screenWidth;
	int screenHeight;
};

extern data gpudata;
