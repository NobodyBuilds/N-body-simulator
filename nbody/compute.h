#pragma once

#ifdef __cplusplus
extern "C" {
#endif
	void registerBodies();

	bool initgpu(int count);
	void  freeDynamicGrid();
	void freegpu();
	void syncstruct();

	bool initDynamicGrid(int totalbodies);

	void computephysics(float dt);
	void registerGLBuffer(unsigned int vboId);
	void unregisterGLBuffer();
	void changestarsize();
	void changestarmass();
	void render();
#ifdef __cplusplus
}
#endif

