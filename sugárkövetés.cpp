//=============================================================================================
// Mintaprogram: Zold háromszog. Ervenyes 2025.-tol
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni, ideértve ChatGPT-t és társait is
// - felesleges programsorokat a beadott programban hagyni 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL es GLM fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Zelch Csaba
// Neptun : LK0617
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================//=============================================================================================
#include "framework.h"

const int windowWidth = 600, windowHeight = 600;

enum MaterialType {
	ROUGH, REFLECTIVE, REFRACTIVE
};

struct Material {
	vec3 ka, kd, ks;
	float  shininess; 
	vec3 F0;
	vec3 n;
	MaterialType type;

	Material(MaterialType t) {
		type = t;
	}

};

struct RoughMaterial :Material {

	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * 3.0f;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}

};

class txt {

	unsigned int textureId = 0;

public:
	txt(int width, int height, std::vector<vec3>& image) {

		glGenTextures(1, &textureId); 
		glBindTexture(GL_TEXTURE_2D, textureId);    
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]); 
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	}
	void Bind(int textureUnit) {
		glActiveTexture(GL_TEXTURE0 + textureUnit); 
		glBindTexture(GL_TEXTURE_2D, textureId); 
	}
	~txt() {
		if (textureId > 0) glDeleteTextures(1, &textureId);
	}
};

vec3 operator/(vec3 a, vec3 b) {
	return vec3(a.x / b.x, a.y / b.y, a.z / b.z);
}


struct ReflectiveMaterial :Material {

	ReflectiveMaterial(vec3 n, vec3 kappa) :Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}

};

struct RefractiveMaterial :Material {

	RefractiveMaterial(vec3 n, vec3 kappa) :Material(REFRACTIVE) {
		vec3 one(1, 1, 1);
		this->n = n;
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}

};


struct Hit {

	float t; 
	vec3 position, normal;
	Material* material;

	Hit() {
		t = -1;
	}

};

struct Ray {

	vec3 start, dir; 
	bool out;

	Ray(vec3 _start, vec3 _dir, bool _out = true) {
		start = _start;
		dir = normalize(_dir);
		out = _out;
	}

};

class Intersectable {

protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0; 

};

class sik {

	float A, B, C, D;

public:
	sik(vec3 pont, vec3 n) {
		A = n.x;
		B = n.y;
		C = n.z;
		D = -(n.x * pont.x + n.y * pont.y + n.z * pont.z);
	}

	float behelyettesit(vec3 pont) {
		return A * pont.x + B * pont.y + C * pont.z + D;
	}

};

class Cylinder : public Intersectable {

	vec3 s;       
	vec3 d;
	float radius, h;

public:
	Cylinder(const vec3& _s, const vec3& _d, float _radius, float _h, Material* _material) {
		h = _h;
		s = _s;
		d = normalize(_d);
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {

		sik sik1(s, d);
		sik sik2(s + h * d, d);

		Hit hit;
		vec3 dir = ray.dir;
		vec3 m = ray.start - s;

		vec3 d_v = dir - d * dot(dir, d);
		vec3 m_v = m - d * dot(m, d);

		//a,b,c meghatározása és normálvektor képletének meghatározása: https://chatgpt.com/
		float a = dot(d_v, d_v);
		float b = 2.0f * dot(d_v, m_v);
		float c = dot(m_v, m_v) - radius * radius;

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;

		float sqrt_discr = sqrtf(discr);

		float t1 = (-b + sqrt_discr) / 2.0f / a;
		vec3 p1 = ray.start + ray.dir * t1;
		if (sik1.behelyettesit(p1) < 0 || sik2.behelyettesit(p1) > 0) t1 = -1;

		float t2 = (-b - sqrt_discr) / 2.0f / a;
		vec3 p2 = ray.start + ray.dir * t2;
		if (sik1.behelyettesit(p2) < 0 || sik2.behelyettesit(p2) > 0) t2 = -1;

		if (t1 <= 0 && t2 <= 0) return hit;
		if (t1 <= 0) hit.t = t2;
		else if (t2 <= 0) hit.t = t1;
		else if (t2 < t1) hit.t = t2;
		else hit.t = t1;


		hit.position = ray.start + ray.dir * hit.t;

		vec3 tmp = hit.position - s;
		vec3 projection = d * dot(tmp, d);
		vec3 normal = normalize(tmp - projection);
		hit.normal = normal;

		hit.material = material;
		return hit;

	}
};

class Cone : public Intersectable {

	vec3 p;
	vec3 d;
	float h;
	float alpha;

public:
	Cone(const vec3& _p, const vec3& _d, float _h, float _alpha, Material* _material) {
		p = _p;
		d = normalize(_d);
		h = _h;
		alpha = _alpha;
		material = _material;
	}

	Hit intersect(const Ray& ray) {

		sik sik1(p, d);
		sik sik2(p + h * d, d);

		Hit hit;
		vec3 m = ray.start - p;
		float cos2 = cosf(alpha) * cosf(alpha);

		float dv = dot(ray.dir, d);
		float mv = dot(m, d);

		//a,b,c meghatározása és normálvektor képletének meghatározása: https://chatgpt.com/
		float a = dv * dv - dot(ray.dir, ray.dir) * cos2;
		float b = 2 * (dv * mv - dot(ray.dir, m) * cos2);
		float c = mv * mv - dot(m, m) * cos2;

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;

		float sqrt_discr = sqrtf(discr);

		float t1 = (-b + sqrt_discr) / 2.0f / a;
		vec3 p1 = ray.start + ray.dir * t1;
		if (sik1.behelyettesit(p1) < 0 || sik2.behelyettesit(p1) > 0) t1 = -1;

		float t2 = (-b - sqrt_discr) / 2.0f / a;
		vec3 p2 = ray.start + ray.dir * t2;
		if (sik1.behelyettesit(p2) < 0 || sik2.behelyettesit(p2) > 0) t2 = -1;

		if (t1 <= 0 && t2 <= 0) return hit;
		if (t1 <= 0) hit.t = t2;
		else if (t2 <= 0) hit.t = t1;
		else if (t2 < t1) hit.t = t2;
		else hit.t = t1;


		hit.position = ray.start + ray.dir * hit.t;

		vec3 normal = normalize(dot(hit.position - p, d) * d - cos2 * (hit.position - p));

		hit.normal = normal;
		hit.material = material;
		return hit;

	}
};

class Plane : public Intersectable {

	vec3 center;
	float size;
	vec3 n;
	Material* material2;

public:
	Plane(vec3 _center, vec3 _n, float _size, Material* _material, Material* _material2) {

		center = _center;
		size = _size;
		n = normalize(_n);
		material = _material;
		material2 = _material2;
	}

	Hit intersect(const Ray& ray) {

		Hit hit;

		float t = dot(center - ray.start, n) / dot(ray.dir, n);

		if (t <= 0) return hit;

		vec3 position = ray.start + ray.dir * t;

		if ((position.x > 11.0f || position.x < -10.0f) || (position.z > 11.0f || position.z < -10.0f)) return hit;

		int ix = static_cast<int>(floor(position.x));
		int iz = static_cast<int>(floor(position.z));

		if ((ix + iz) % 2 == 0) {
			hit.material = material;
		}
		else {
			hit.material = material2;
		}

		hit.t = t;
		hit.position = position;
		hit.normal = n;

		return hit;

	}

};

class Camera {

	vec3 eye, lookat, right, up; 
	float fov; 

public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) { 
		eye = _eye; lookat = _lookat; fov = _fov;
		vec3 w = eye - lookat;
		float windowSize = length(w) * tanf(fov / 2); 
		right = normalize(cross(vup, w)) * (float)windowSize * (float)windowWidth / (float)windowHeight;
		up = normalize(cross(w, right)) * windowSize;
	}

	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2 * (X + 0.5f) / windowWidth - 1) + up * (2 * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	void Animate() { 
		vec3 d = eye - lookat;

		float radius = length(d);
		float angle = atan2(d.z, d.x);
		angle += M_PI / 4.0f;

		vec3 newDir = vec3(cos(angle), 0, sin(angle)) * radius;

		eye = lookat + vec3(newDir.x, d.y, newDir.z);
		vec3 newUp = vec3(0, 1, 0);
		set(eye, lookat, newUp, fov);
	}

};

struct Light { 

	vec3 direction; 
	vec3 Le;

	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

const float Epsilon = 0.0001f;


class Scene {

	std::vector<Intersectable*> objects; 
	std::vector<Light*> lights; 
	Camera camera; 
	vec3 La; 

public:
	void build() { 

		vec3 eye = vec3(0.0f, 1.0f, 4.0f), vup = vec3(0.0f, 1.0f, 0.0f), lookat = vec3(0.0f, 0.0f, 0.0f);
		float fov = 45.0f * (float)M_PI / 180.0f;
		camera.set(eye, lookat, vup, fov); 

		La = vec3(0.4f, 0.4f, 0.4f); 
		vec3 lightDirection(1.0f, 1.0f, 1.0f), Le(2.0f, 2.0f, 2.0f);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd1(0.1f, 0.2f, 0.3f);
		vec3 kd2(0.3f, 0.0f, 0.2f);
		vec3 kd3(0.3f, 0.2f, 0.1f);
		vec3 kd4(0, 0.1f, 0.3f);
		vec3 kd5(0.3f, 0.3f, 0.3f);
		vec3 ks(2.0f, 2.0f, 2.0f);
		vec3 ks2(0.0f, 0.0f, 0.0f);

		vec3 n(0.17f, 0.35f, 1.5f), kappa(3.1f, 2.7f, 1.9f);
		vec3 n2(1.3f, 1.3f, 1.3f), kappa2(0.0f, 0.0f, 0.0f);

		Material* cian = new RoughMaterial(kd1, ks, 100);
		Material* magenta = new RoughMaterial(kd2, ks, 20);
		Material* sarga = new RoughMaterial(kd3, ks, 50);
		Material* kek = new RoughMaterial(kd4, ks2, 0.0f);
		Material* feher = new RoughMaterial(kd5, ks2, 0.0f);

		Material* gold = new ReflectiveMaterial(n, kappa);
		Material* water = new RefractiveMaterial(n2, kappa2);

		objects.push_back(new Cylinder(vec3(1, -1, 0), vec3(0.1, 1, 0), 0.3, 2, gold));

		objects.push_back(new Cylinder(vec3(0, -1, -0.8), vec3(-0.2, 1, -0.1), 0.3, 2, water));

		objects.push_back(new Cylinder(vec3(-1, -1, 0), vec3(0, 1, 0.1), 0.3, 2, sarga));

		objects.push_back(new Cone(vec3(0, 1, 0), vec3(-0.1, -1, -0.05), 2, 0.2, cian));

		objects.push_back(new Cone(vec3(0, 1, 0.8), vec3(0.2, -1, 0), 2, 0.2, magenta));

		objects.push_back(new Plane(vec3(0, -1, 0), vec3(0, 1, 0), 20.0f, feher, kek));
	}

	void render(std::vector<vec3>& image) { 
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y)); 
				image[Y * windowWidth + X] = vec3(color.x, color.y, color.z);
			}
		}
	}

	Hit firstIntersect(Ray ray) { 

		Hit bestHit;

		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); 
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}

		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = -bestHit.normal; 
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) { 

		if (depth > 5) {
			return La;
		}

		Hit hit = firstIntersect(ray); 
		if (hit.t < 0) return La; 

		vec3 outRadiance(0, 0, 0);

		if (hit.material->type == ROUGH) {

			outRadiance = hit.material->ka * La;

			for (Light* light : lights) { 

				Ray shadowRay(hit.position + hit.normal * Epsilon, light->direction);
				float cosTheta = dot(hit.normal, light->direction); 

				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {

					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta; 
					vec3 halfway = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfway);

					if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);

				}
			}
		}

		if (hit.material->type == REFLECTIVE) {

			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;

			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * powf(1 - cosa, 5);

			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * Epsilon, reflectedDir), depth + 1) * F;

		}

		if (hit.material->type == REFRACTIVE) {

			float ior = (ray.out) ? hit.material->n.x : 1 / hit.material->n.x;
			vec3 refractionDir = refract(ray.dir, hit.normal, ior);

			if (length(refractionDir) > 0) {

				float cosa = -dot(ray.dir, hit.normal);
				vec3 one(1, 1, 1);
				vec3 F = hit.material->F0 + (one - hit.material->F0) * powf(1 - cosa, 5);

				outRadiance = outRadiance + trace(Ray(hit.position - hit.normal * Epsilon, refractionDir, !ray.out), depth + 1) * (vec3(1, 1, 1) - F);
			}
		}

		return outRadiance; 

	}

	vec3 refract(vec3 v, vec3 n, float ior) {

		float cosa = -dot(v, n);
		float disc = 1 - (1 - cosa * cosa) / ior / ior;
		if (disc < 0) return vec3(0, 0, 0);
		return v / ior + n * (cosa / ior - sqrtf(disc));

	}

	void Animate() { camera.Animate(); }
};


Scene scene;

GPUProgram gpuProgram;


const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	
	layout(location = 1) in vec2 vertexUV;

	out vec2 texcoord; //kapott csucshoz tartozo textura koordinata

	void main() {
		texcoord = vertexUV;
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		
	}
)";

const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			
	out vec4 fragmentColor;		

	void main() { fragmentColor = texture(textureUnit, texcoord); } 
)";

class FullScreenTexturedQuad : public Geometry<vec2> {

	unsigned int vao = 0, vbo[2];
	txt* texture;
	unsigned int textureId = 0;

public:
	FullScreenTexturedQuad() {

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(2, &vbo[0]);  

		vtx = { vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, 1) };

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, vtx.size() * sizeof(vec2), &vtx[0], GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		std::vector<vec2> uvs = { vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1) };
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(vec2), &uvs[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

	}

	void LoadTexture(int width, int height, std::vector<vec3>& image) {
		texture = new txt(width, height, image);
	}

	void Draw(GPUProgram* gpuProgram) { 
		int textureUnit = 0;
		gpuProgram->setUniform(textureUnit, "textureUnit");
		texture->Bind(textureUnit);
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}

};


class RaytraceApp : public glApp {

	GPUProgram* gpuProgram;
	FullScreenTexturedQuad* fullScreenTexturedQuad;

public:
	RaytraceApp() : glApp("Ray tracing") {}

	void onInitialization() {

		glViewport(0, 0, windowWidth, windowHeight);
		gpuProgram = new GPUProgram(vertexSource, fragmentSource);
		scene.build();
		fullScreenTexturedQuad = new FullScreenTexturedQuad;

	}

	void onDisplay() {

		std::vector<vec3> image(windowWidth * windowHeight);
		scene.render(image);
		fullScreenTexturedQuad->LoadTexture(windowWidth, windowHeight, image);
		fullScreenTexturedQuad->Draw(gpuProgram);

	}

	void onKeyboard(int key) {

		if (key == 'a') {

			scene.Animate();
			refreshScreen();

		}
	}
} app;