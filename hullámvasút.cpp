#include "../framework.h"

// csucspont arnyalo
const char* vertSource = R"(
	#version 330		

	uniform mat4 MVP;

	layout(location = 0) in vec2 vertexPosition;
	void main() {
		gl_Position = MVP * vec4(vertexPosition, 0, 1);
	}

)";

// pixel arnyalo
const char* fragSource = R"(
	#version 330

	uniform vec3 color;
	out vec4 fragmentColor;
	void main() {
		fragmentColor = vec4(color, 1);
	}
)";

//kamera osztaly
class Camera {

	vec2 wCenter; // center in world coords
	vec2 wSize; // width and height in world coords

public:

	Camera() : wCenter(0, 0), wSize(20, 20) {}

	mat4 V() { return translate(vec3(-wCenter.x, -wCenter.y, 0)); }

	mat4 P() { // projection matrix
		return scale(vec3(2 / wSize.x, 2 / wSize.y, 1));
	}
	mat4 Vinv() { // inverse view matrix
		return translate(vec3(wCenter.x, wCenter.y, 0));
	}
	mat4 Pinv() { // inverse projection matrix
		return scale(vec3(wSize.x / 2, wSize.y / 2, 1));
	}

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }

	vec2 ScreentoWorld(float cX, float cY) {


		vec4 mVertex = Vinv() * Pinv() * vec4(cX, cY, 0, 1);
		return vec2(mVertex.x, mVertex.y);

	}
};

class Spline :public Geometry<vec2> {

	std::vector<vec2> cps; 
	std::vector<float> ts; 
	float maxTau;

public:

	float getmaxTau() {
		return maxTau;
	}

	vec2 Hermite(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t) { 

		t -= t0;
		float dt = t1 - t0;
		vec2 a0 = p0, a1 = v0;
		vec2 a2 = (p1 - p0) * 3.0f / dt / dt - (v1 + v0 * 2.0f) / dt;
		vec2 a3 = (p0 - p1) * 2.0f / dt / dt / dt + (v1 + v0) / dt / dt;
		return ((a3 * t + a2) * t + a1) * t + a0; 

	}

	vec2 Derivate(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t) {

		t -= t0;
		float dt = t1 - t0;
		vec2 a1 = v0;
		vec2 a2 = (p1 - p0) * 3.0f / dt / dt - (v1 + v0 * 2.0f) / dt;
		vec2 a3 = (p0 - p1) * 2.0f / dt / dt / dt + (v1 + v0) / dt / dt;
		return (3.0f * a3 * t + 2.0f * a2) * t + a1;

	}

	vec2 Derivate2x(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t) {

		t -= t0;
		float dt = t1 - t0;
		vec2 a2 = (p1 - p0) * 3.0f / dt / dt - (v1 + v0 * 2.0f) / dt;
		vec2 a3 = (p0 - p1) * 2.0f / dt / dt / dt + (v1 + v0) / dt / dt;
		return 6.0f * a3 * t + 2.0f * a2;

	}

	vec2 CalculateTangent(unsigned i) {
		if (i == 0) {

			return vec2(0, 0);

		}
		else if (i == cps.size() - 1) {

			return vec2(0, 0);

		}else {

			return (1.0f / 2.0f) * (((cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i])) + ((cps[i] - cps[i - 1]) / (ts[i] - ts[i - 1])));

		}
	}

	vec2 r(float t) {

		for (unsigned i = 0; i < cps.size() - 1; i++) {

			if (ts[i] <= t && t <= ts[i + 1]) {

				vec2 v0 = CalculateTangent(i);
				vec2 v1 = CalculateTangent(i + 1);


				return Hermite(cps[i], v0, ts[i], cps[i + 1], v1, ts[i + 1], t);

			}
		}
		return vec2(0, 0);
	}

	vec2 v(float t) {

		for (unsigned i = 0; i < cps.size() - 1; i++) {

			if (ts[i] <= t && t <= ts[i + 1]) {

				vec2 v0 = CalculateTangent(i);
				vec2 v1 = CalculateTangent(i + 1);

				return Derivate(cps[i], v0, ts[i], cps[i + 1], v1, ts[i + 1], t);

			}
		}
		return vec2(0,0);
	}

	vec2 a(float t) {

		for (unsigned i = 0; i < cps.size() - 1; i++) {

			if (ts[i] <= t && t <= ts[i + 1]) {

				vec2 v0 = CalculateTangent(i);
				vec2 v1 = CalculateTangent(i + 1);

				return Derivate2x(cps[i], v0, ts[i], cps[i + 1], v1, ts[i + 1], t);

			}
		}
		return vec2(0, 0);
	}

	float vektorhossz(vec2 v) {

		return sqrt(v.x * v.x + v.y * v.y);

	}

	vec2 T(float t) {

		return v(t) / vektorhossz(v(t));

	}

	vec2 N(float t) {

		return vec2(-T(t).y, T(t).x);

	}

	void AddControlPoint(vec2 cp) {

		float ti = cps.size(); 
		maxTau = cps.size();
		cps.push_back(cp); 
		ts.push_back(ti);
		update();

	}

	void update() {

		if (cps.size() > 1) {

			Vtx().clear();
			int numPoints = 100; // Pontosan 100 pontot akarunk
			float tMin = ts.front();  
			float tMax = ts.back();   

			for (int i = 0; i < numPoints; i++) {

				float t = tMin + (tMax - tMin) * (float(i) / (numPoints - 1));
				Vtx().push_back(r(t)); 

			}

			updateGPU();

		}
	}

	void DrawSpline(GPUProgram* gpuProgram, vec3 color, Camera& camera) {

		if (cps.size() > 1) {

			mat4 MVP = camera.P() * camera.V();
			gpuProgram->setUniform(MVP, "MVP");
			Draw(gpuProgram, GL_LINE_STRIP, color);

		}
	}


	void DrawPoints(GPUProgram* gpuProgram, vec3 color, Camera& camera) {

		if (cps.size() > 0) {

			Geometry<vec2> points;

			for (unsigned i = 0; i < cps.size(); i++) {

				points.Vtx().push_back(cps[i]);

			}

			points.updateGPU();
			mat4 MVP = camera.P() * camera.V();
			gpuProgram->setUniform(MVP, "MVP");
			points.Draw(gpuProgram, GL_POINTS, color);

		}	
	}
};

enum State {
	VARAKOZIK,
	INDITOTT,
	LEESETT
};

class Circle : public Geometry<vec2> {

	vec2 wTranslate;	// translation
	float phi;			// angle of rotation
	State state;
	Spline* spline;
	float Tau;
	vec2 vel;
	vec2 pos;
	float szogsebesseg;

public:

	Circle(Spline* sp) {

		const int nVertices = 100; // soksz�g cs�csainak sz�ma
		for (unsigned i = 0; i < nVertices; i++) {

			float phil = i * 2.0f * (float)M_PI / nVertices;
			Vtx().push_back(vec2(cosf(phil), sinf(phil))); // k�r egyenlete

		}

		updateGPU(); // CPU->GPU szinkroniz�l�s
		state = VARAKOZIK;
		spline = sp;
		Tau = 0.01;
		phi = 0;
		wTranslate = vec2(0,0);
		szogsebesseg = 0;
		pos = vec2(0, 0);
		vel = vec2(0, 0);

	}

	void setState(State st) {
		state = st;
	}

	State getState() {
		return state;
	}

	 float velocity() {
		 float numb = ((2 * 40 * (spline->r(0.001f).y - spline->r(Tau).y)) / (1.0f + 0.5f));
		 if (numb >= 0) {
			 return sqrt(numb);
		 }
		 else {
			 return numb;
		 }
	}

	 float kappa(float Tau) {
		 return (dot(spline->a(Tau), spline->N(Tau)) / (spline->vektorhossz(spline->v(Tau)) * spline->vektorhossz(spline->v(Tau))));
	}

	 float K() {
		 vec2 g = vec2(0, 40);
		 return (dot(g, spline->N(Tau)) + ((velocity() * velocity()) * kappa(Tau)));
	}

	 void beallitkozeppont(float Tau) {
		 wTranslate = spline->r(Tau) + spline->N(Tau);
	}

	mat4 M() {

		mat4 Mrotate(cosf(phi), sinf(phi), 0, 0,
			-sinf(phi), cosf(phi), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1); // rotation

		mat4 Mtranslate(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTranslate.x, wTranslate.y, 0, 1); // translation

		return Mtranslate * Mrotate;	// model transformation
	}

	void Start() {

		Tau = 0.01f;
		//phi = atan(spline->v(Tau).y/spline->v(Tau).x);
		phi = 0;
		beallitkozeppont(Tau);
		state = INDITOTT;

	}

	void Animate(float Dt) {

		if (state == INDITOTT) {

			float v = velocity();

			if ((v >= 0)) {

				if (K() >= 0) {

					float deltaTau = ((v * Dt) / spline->vektorhossz(spline->v(Tau)));

					if ((Tau + deltaTau) <= spline->getmaxTau()) {

						vec2 position = spline->r(Tau) + spline->N(Tau);
						vec2 newposition = spline->r(Tau + deltaTau) + spline->N(Tau + deltaTau);

						float distance = sqrt((newposition.x - position.x) * (newposition.x - position.x) + (newposition.y - position.y) * (newposition.y - position.y));

						phi -= distance;
						szogsebesseg = fabs(distance)/Dt;
						//phi = atan(spline->v(Tau).y / spline->v(Tau).x);
						Tau = Tau + deltaTau;
						beallitkozeppont(Tau);
					}
					else {

						state = LEESETT;
						vel = v * spline->T(Tau);
						pos = spline->r(Tau) + spline->N(Tau);
						vec2 g = vec2(0, -40);
						vel = vel + g * Dt;
						pos = pos + vel * Dt + (1.0f / 2.0f) * g * (Dt * Dt);
						wTranslate = pos;
						phi -= szogsebesseg* Dt;

					}
				}
				else {

					state = LEESETT;
					vel = v * spline->T(Tau);
					pos = spline->r(Tau) + spline->N(Tau);
					vec2 g = vec2(0, -40);
					vel = vel + g * Dt;
					pos = pos + vel * Dt + (1.0f / 2.0f) * g * (Dt * Dt);
					wTranslate = pos;
					phi -= szogsebesseg* Dt;
				}
			}
			else {

				Start();

			}


		}
		else if (state == LEESETT) {
			vec2 g = vec2(0, -40);
			vel = vel + g * Dt;
			pos = pos + vel * Dt + (1.0f / 2.0f) * g * (Dt * Dt);
			wTranslate = pos;
			phi -= szogsebesseg* Dt;
		}
	}

	void DrawCircle(GPUProgram* gpuProgram, vec3 color, Camera& camera) {

		mat4 MVP = camera.P() * camera.V() * M();
		gpuProgram->setUniform(MVP, "MVP");
		Draw(gpuProgram, GL_TRIANGLE_FAN, color);

	}

	void DrawKullok(GPUProgram* gpuProgram, vec3 color, Camera& camera) {

		Geometry<vec2> korlap;
		Geometry<vec2> kullo1;
		Geometry<vec2> kullo2;
		for (unsigned i = 0; i < Vtx().size();i++) {

			korlap.Vtx().push_back(Vtx()[i]);

		}
		kullo1.Vtx().push_back(Vtx()[0]);
		kullo1.Vtx().push_back(Vtx()[50]);
		kullo2.Vtx().push_back(Vtx()[25]);
		kullo2.Vtx().push_back(Vtx()[75]);

		korlap.updateGPU();
		kullo1.updateGPU();
		kullo2.updateGPU();

		mat4 MVP = camera.P() * camera.V() * M();
		gpuProgram->setUniform(MVP, "MVP");
		korlap.Draw(gpuProgram, GL_LINE_LOOP, color);
		kullo1.Draw(gpuProgram, GL_LINES, color);
		kullo2.Draw(gpuProgram, GL_LINES, color);

	}
};

const int winWidth = 600, winHeight = 600;

class MyApp : public glApp {

	GPUProgram* gpuProgram;	   
	Camera* camera;
	Spline* spline;
	Circle* circle;
	float ok;

public:

	MyApp() : glApp("Green triangle") {}

	void onInitialization() {

		glPointSize(10);
		glLineWidth(3);

		spline = new Spline();
		camera = new Camera();
		circle = nullptr;
		gpuProgram = new GPUProgram(vertSource, fragSource);

	}

	void onDisplay() {

		glClearColor(0, 0, 0, 0);    
		glClear(GL_COLOR_BUFFER_BIT); 
		glViewport(0, 0, winWidth, winHeight);
		if (circle != nullptr) {

			circle->DrawCircle(gpuProgram, vec3(0, 0, 1), *camera);
			circle->DrawKullok(gpuProgram, vec3(1, 1, 1), *camera);

		}
		spline->DrawSpline(gpuProgram, vec3(1, 1, 0), *camera);
		spline->DrawPoints(gpuProgram, vec3(1, 0, 0), *camera);
		

	}

	void onKeyboard(int key) { 

		if (key == ' ') {

			if (circle == nullptr) {

				circle = new Circle(spline);
				circle->Start();
				ok = 0.0f;
				refreshScreen();

			}else {

				circle->Start();
				ok = 0.0f;
				refreshScreen();

			}
		}
	}

	void onMousePressed(MouseButton button, int pX, int pY) {

		if (button == MOUSE_LEFT) {

			float cX = 2.0f * pX / winWidth - 1;	// flip y axis
			float cY = 1.0f - 2.0f * pY / winHeight;
			vec2 worldPoint = camera->ScreentoWorld(cX, cY);
			spline->AddControlPoint(worldPoint);
			refreshScreen();     // redraw

		}
	}

	void onTimeElapsed(float startTime, float endTime) {

		if (circle != nullptr && circle->getState() == INDITOTT) {

			const float dt = 0.01;
			for (float t = startTime; t < endTime; t += dt) {

				float Dt = fmin(dt, endTime - t);
				circle->Animate(Dt);
			}

			refreshScreen(); 

		}
		else if (circle != nullptr && circle->getState() == LEESETT) {
			const float dt = 0.01;
			for (float t = startTime; t < endTime; t += dt) {

				float Dt = fmin(dt, endTime - t);
				ok += Dt;
				circle->Animate(Dt);
			}

			
			if (ok > 3.0f) {
				circle = nullptr;
			}
			refreshScreen();
		}
	}
}app;