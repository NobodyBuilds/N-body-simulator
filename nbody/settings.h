#pragma once
struct param {
	// === FLOAT VARIABLES (4 bytes each) ===
	// Grouped for cache locality and alignment
	float fixedDt = 1 / 120.0f;
	float simspeed = 1.0f;
	float size = 1.0f;
	float particleMass = 1.0f;
	float cold = 2.00f;
	float heatMultiplier = 12.0f;
	float h = 2.50f;
	float h2 = h * h;
	float pressure = 1200.0f;
	float pi = 3.14159265358979323846f;
	float pollycoef6 = 0.0f;
	float spikycoef = 0.0f;
	float Sdensity = 0.0f;
	float cellSize = 1.0f;
	float spikygradv = 0.0f;
	float wx = 0.0f;
	float wy = 0.0f;
	float wz = 0.0f;
	float avgFps = 0.0f;
	float minFps = 1000.0f;
	float maxFps = 0.0f;
	float fpsTimer = 0.0f;
	float accumulator = 0.0f;
	float fps = 0.0f;
	float centermass = 10000.0f;
	float starsize = 15.0f;
	float radius = 200.0f;
	float maxradius = 500.0f;
	float orbitspeed = 1.0f;
	float G = 6.67f;
	float dst = 250.0f;
	float impactspeed = 6.0f;
	float yspeed = 0.5f;
	float min_density, max_density, avg_density = 0;
	float min_neardensity, max_neardensity, avg_neardensity = 0;
	float gamma = 5.0f;//7/5,2-7,5/3
	float alpha_v = 0.02f;
	float beta_v = 0.04f;
	float restdensity = 0.15f;
	float epsilon = 0.03f;
	float nearpressure = 200.0f;
	float ndensity = 0.0f;

	double fuc_ms = 0.0;
	// === INT VARIABLES (4 bytes each) ===
	int totalBodies = 10000;
	int count = totalBodies;
	int min_n, max_n, avg_n = 0;
	int substeps = 1;
	int fpsCount = 0;
	int mode = 0;
	int screenWidth;
	int screenHeight;

	// === BOOL VARIABLES (1 byte each) ===
	bool sph = true;
	bool gravity = true;
	bool nopause = true;
	bool heateffect = true;
	bool recordSim = false;
	bool gui = true;
	bool lockstar = false;
	bool firstframe = true;
};
extern param settings;

struct data {
	float dt;
	float restdensity;
	float epsilon;
	float nearpressure;
	float ndensity;
	float particlemass;
	float h;
	float spacing;
	float sdensity;
	float h2;
	float pollycoef6;
	float spikycoef;
	float pressure;
	float spikyGradv;
	float G;
	float orbitalspeed;
	float radius;
	float centermass;
	float starsize;
	float maxradius;
	float dst;
	float impactspeed;
	float yspeed;
	float particlesize;
	float gamma;
	float alpha_v;
	float beta_v;
	int mode;
	int count;
	int screenWidth;
	int screenHeight;
	bool lockstar;
};

extern data gpudata;
