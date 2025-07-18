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

//a megoldáshoz felhasználtam a tárgy honlapján a 3d inkrementális képszintézisnél feltüntetett 3D motorka programot https://www.youtube.com/watch?v=HNJRpNPGETs&ab_channel=LaszloSzirmay-Kalos

const char * vertSource = R"(
	 #version 330      
        
        uniform mat4  MVP, M, Minv; 
  
        uniform vec4 wLiPos;
        uniform vec3  wEye;           
        
        layout(location = 0) in vec3  vtxPos;       
        layout(location = 1) in vec3  vtxNorm;
        layout (location = 2) in vec2 vtxUV;     
        
        out vec3 wNormal; 
        out vec3 wView; 
        out vec3 wLight;  
        
        out vec4 wPos;

        out vec2 texcoord; 

        void main() {

            gl_Position = MVP * vec4(vtxPos, 1); 
            
            wPos =  M * vec4(vtxPos, 1);

            wLight = wLiPos.xyz * wPos.w - wPos.xyz * wLiPos.w;
            wView = wEye - wPos.xyz/wPos.w;
            wNormal = (vec4(vtxNorm, 0)*Minv).xyz;

            texcoord = vtxUV;
            
        }
    
)";

const char * fragSource = R"(
	#version 330

    uniform int triangleCount;
    uniform vec3 triangles[186];

    uniform vec3 kd, ks, ka;
    uniform float shine;

    uniform vec3 La, Le;

    uniform sampler2D textureUnit;
    
    in vec3 wNormal; 
    in vec3 wView; 
    in vec3 wLight;

    in vec4 wPos;

    uniform bool useTexture;
    in  vec2 texcoord;	
	
	out vec4 fragmentColor;	

    bool IntersectsTriangle(vec3 rayOrig, vec3 rayDir, vec3 r1, vec3 r2, vec3 r3) {

        float EPSILON = 0.001;
        vec3 n = cross(r2 - r1, r3 - r1);

        if(abs(dot(rayDir, n)) < EPSILON){
            return false;
        }

        float t = dot(r1-rayOrig,n)/dot(rayDir, n);

        if(t<0){
            return false;
        }

        vec3 p = rayOrig + t*rayDir;

        vec3 c;

        vec3 edge0 = r2 - r1;
        vec3 vp0 = p - r1;
        c = cross(edge0, vp0);
        if (dot(n, c) < 0) return false;

        vec3 edge1 = r3 - r2;
        vec3 vp1 = p - r2;
        c = cross(edge1, vp1);
        if (dot(n, c) < 0) return false;

        vec3 edge2 = r1 - r3;
        vec3 vp2 = p - r3;
        c = cross(edge2, vp2);
        if (dot(n, c) < 0) return false;

        return true;

    }	

	void main() {

        vec3 N = normalize(wNormal);
        vec3 V = normalize(wView);
        vec3 L = normalize(wLight);
        vec3 H = normalize(L + V);

        bool inShadow = false;

        vec3 ro = wPos.xyz+vec3(0.01*N);
        vec3 rd = L;

        for (int i = 0; i < triangleCount; i++) {

            vec3 r1 = triangles[i * 3 + 0].xyz;
            vec3 r2 = triangles[i * 3 + 1].xyz;
            vec3 r3 = triangles[i * 3 + 2].xyz;

            if (IntersectsTriangle(ro, rd, r1, r2, r3)) {
                inShadow = true;
                break;
            }
        }

        float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);

        vec3 color = vec3(0,0,0);
        vec3 texColor;

        if(useTexture){

            if(inShadow){

                texColor = texture(textureUnit, texcoord).rgb;
                vec3 nka = ka * texColor;
                color = nka*La;

            }else{

                texColor = texture(textureUnit, texcoord).rgb;
                vec3 nka = ka * texColor;
                vec3 nkd = kd* texColor;
                color = nka * La + (nkd * cost + ks * pow(cosd,shine)) * Le;

            }

        }else{

            if(inShadow){

                color = ka * La;

            }else{

                color = ka * La + (kd * cost + ks * pow(cosd,shine)) * Le;

            }
        }

		fragmentColor = vec4(color,1); 

	}
)";

const int winWidth = 600, winHeight = 600;

class CheckerBoardTexture : public Texture {
   
public:

    CheckerBoardTexture(const int width, const int height) : Texture() {

        std::vector<vec3> image(width * height);
        const vec3 white(0.3, 0.3, 0.3), blue(0, 0.1, 0.3);

        for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
            image[y * width + x] = (x & 1) ^ (y & 1) ? blue : white;
        }

        updateTexture(width, height, image);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    }
};

struct Dnum { 

    float f; 
    vec2 d;

    Dnum(float f0 = 0, vec2 d0 = vec2(0)) { f = f0, d = d0; }

    Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
    Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }

    Dnum operator*(Dnum r) {
        return Dnum(f * r.f, f * r.d + d * r.f);
    }

    Dnum operator/(Dnum r) {
        return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
    }
};

Dnum Sin(Dnum g) { return Dnum(sinf(g.f), cosf(g.f) * g.d); }
Dnum Cos(Dnum g) { return Dnum(cosf(g.f), -sinf(g.f) * g.d); }

mat4 TranslateMatrix(vec3 e) {
    mat4 result = mat4(1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        e.x, e.y, e.z, 1);

    return result;
}

mat4 ScaleMatrix(vec3 s) {
    mat4 result = mat4(s.x, 0, 0, 0,
        0, s.y, 0, 0,
        0, 0, s.z, 0,
        0, 0, 0, 1);

    return result;
}

mat4 RotationMatrix(float rAn, vec3 rAx) {
    vec3 r1 = vec3(1, 0, 0);
    vec3 r2 = vec3(0, 1, 0);
    vec3 r3 = vec3(0, 0, 1);
    vec3 i = r1 * cosf(rAn) + rAx * (dot(r1, rAx)) * (1 - cosf(rAn)) + cross(rAx, r1) * sinf(rAn);
    vec3 j = r2 * cosf(rAn) + rAx * (dot(r2, rAx)) * (1 - cosf(rAn)) + cross(rAx, r2) * sinf(rAn);
    vec3 k = r3 * cosf(rAn) + rAx * (dot(r3, rAx)) * (1 - cosf(rAn)) + cross(rAx, r3) * sinf(rAn);
    mat4 result = mat4(i.x, j.x, k.x, 0,
        i.y, j.y, k.y, 0,
        i.z, j.z, k.z, 0,
        0, 0, 0, 1);
    return result;
}

struct Camera { 

    vec3 wEye, wLookat, wUp; 
    float fov, asp, fp, bp; 

public:
    Camera() {
        asp = (float)winWidth / winHeight;
        fov = 45.0f * (float)M_PI / 180.0f;
        fp = 1; bp = 20;
    }
     
    mat4 V() { 
        vec3 w = normalize(wEye - wLookat);
        vec3 u = normalize(cross(wUp, w));
        vec3 v = cross(w, u);
        return mat4(u.x, v.x, w.x, 0,
            u.y, v.y, w.y, 0,
            u.z, v.z, w.z, 0,
            0, 0, 0, 1) * TranslateMatrix(wEye * (-1));
    }
    
    mat4 P() { 
        return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
            0, 1 / tan(fov / 2), 0, 0,
            0, 0, -(fp + bp) / (bp - fp), -1,
            0, 0, -2 * fp * bp / (bp - fp), 0);
    }

    void Animate() {
        
        vec3 d = wEye - wLookat;

        float radius = length(d);
        float angle = atan2(d.z, d.x);
        angle += M_PI / 4.0f;

        vec3 newDir = vec3(cos(angle), 0, sin(angle)) * radius;

        float x = cosf(M_PI / 4.0f) * wEye.x - sinf(M_PI / 4.0f) * wEye.z;
        float z = sinf(M_PI / 4.0f) * wEye.x + cosf(M_PI / 4.0f) * wEye.z;

        wEye = vec3(x, wEye.y,z);

    }
    
};

struct Material {

    vec3 kd, ks, ka;
    float shine;
    Material(vec3 _kd, vec3 _ks, vec3 _ka, float _shine) {
        kd = _kd;
        ks = _ks;
        ka = _ka;
        shine = _shine;
    }

};

struct Light {
    
    vec3 La, Le;
    vec4 wLiPos; 

};

struct RenderState {

    mat4 V, P;
    Material* material;
    Light light;
    vec3 wEye;
    int size;
    vec3 triangles[186];

};

struct VtxData {

    vec3 pos, normal;
    vec2 texcoord;

};

template<class T> class Geo {

    unsigned int vao, vbo; 
    std::vector<T> vtx; 

public:
    Geo() {

        glGenVertexArrays(1, &vao); glBindVertexArray(vao);
        glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo); glEnableVertexAttribArray(0);
        int nf = sizeof(T) / sizeof(float);
        if (nf <= 4) glVertexAttribPointer(0, nf, GL_FLOAT, GL_FALSE, 0, NULL);

    }

    std::vector<T>& Vtx() { return vtx; }

    void updateGPU() {

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vtx.size() * sizeof(T), &vtx[0], GL_DYNAMIC_DRAW);

    }

    void Bind() { glBindVertexArray(vao); }

    virtual ~Geo() { glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao); }

};

class Object3D : public Geo<VtxData> {

    int nVtxInStrip, nStrips;

public:
    Object3D() {

        glEnableVertexAttribArray(0); 
        glEnableVertexAttribArray(1); 
        glEnableVertexAttribArray(2); 
        int nb = sizeof(VtxData);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, nb, (void*)offsetof(VtxData, pos));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, nb, (void*)offsetof(VtxData, normal));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, nb, (void*)offsetof(VtxData, texcoord));

    }

    virtual void eval(Dnum& U, Dnum& V, Dnum& X, Dnum& Y, Dnum& Z) = 0;

    VtxData GenVtxData(float u, float v) {

        VtxData vtxData;
        vtxData.texcoord = vec2(u, v);

        Dnum X, Y, Z;
        Dnum U(u, vec2(1, 0)), V(v, vec2(0, 1));
        
        eval(U, V, X, Y, Z);
        
        vtxData.pos = vec3(X.f, Y.f, Z.f);

        vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);

        vtxData.normal = cross(drdU, drdV);

        return vtxData;

    }

    VtxData GenVtxDataPlane(float u, float v) {

        VtxData vtxData;
        vtxData.texcoord = vec2(u, v);

        Dnum X, Y, Z;
        Dnum U(u, vec2(1, 0)), V(v, vec2(0, 1));

        eval(U, V, X, Y, Z);

        vtxData.pos = vec3(X.f, Y.f, Z.f);

        vtxData.normal = vec3(0,1,0);

        return vtxData;

    }
    

    void create(int M, int N, bool isPlane) {
        nVtxInStrip = (M + 1) * 2;
        nStrips = N;

        if (!isPlane) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j <= M; j++) {
                    Vtx().push_back(GenVtxData((float)j / M, (float)i / N));
                    Vtx().push_back(GenVtxData((float)j / M, (float)(i + 1) / N));
                }
            }
        }
        else {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j <= M; j++) {
                    Vtx().push_back(GenVtxDataPlane((float)j / M, (float)i / N));
                    Vtx().push_back(GenVtxDataPlane((float)j / M, (float)(i + 1) / N));
                }
            }
        }
        updateGPU();
    }

    
    void DrawGeometry() {
        Bind();
        for (unsigned int i = 0; i < nStrips; i++)
            glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxInStrip, nVtxInStrip);
    }
};

class Cylinder : public Object3D {

public:
    Cylinder() {
        create(6,1, false);
    }
    void eval(Dnum& U, Dnum& V, Dnum& X, Dnum& Y, Dnum& Z) {

        U = U * 2.0f * M_PI, V = V * 2;
        X = Cos(U); Y = Sin(U); Z = V;
    }

};

class Cone : public Object3D {
public: 
    Cone() {
        create(6, 1, false);
    }
    void eval(Dnum& U, Dnum& V, Dnum& X, Dnum& Y, Dnum& Z) {
        
        U = U * 2.0f * M_PI, V = V * 2.0f;
        Dnum r = V * tanf(0.2f);


        X = Cos(U)* r; Y = Sin(U)* r; Z = V ;

    }
};

class Plane : public Object3D {
public:
    Plane() {
        create(1, 1, true);
    }
    void eval(Dnum& U, Dnum& V, Dnum& X, Dnum& Y, Dnum& Z) {

        X = U-0.5f; Y = -1; Z = V-0.5f;

    }
};

struct Object {

    Material* material;
    Object3D* geometry;
    Texture* texture;

    vec3 scale, translation, rotationAxis;
    float rotationAngle;

public:
    Object(Material* _material, Object3D* _geometry, Texture* _texture) :
        scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
        material = _material;
        geometry = _geometry;
        texture = _texture;
    }

    void SetModelingTransformM(mat4& M) {

        M = TranslateMatrix(translation) * RotationMatrix(rotationAngle, rotationAxis) * ScaleMatrix(scale);

    }

    void SetModelingTransformMinv(mat4& Minv) {

        Minv = ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z)) * RotationMatrix(-rotationAngle, rotationAxis) * TranslateMatrix(-translation);
    }

    void DrawObject(RenderState state, GPUProgram* gpuProgram) {
        mat4 M, Minv;
        SetModelingTransformM(M);
        SetModelingTransformMinv(Minv);
        mat4 MVP = state.P* state.V*M ;

        gpuProgram->setUniform(state.size, "triangleCount");

        gpuProgram->setUniform(MVP, "MVP");
        gpuProgram->setUniform(M, "M");
        gpuProgram->setUniform(Minv, "Minv");

        if (texture != nullptr) {

            int textureUnit = 0;
            gpuProgram->setUniform(textureUnit, "textureUnit");
            texture->Bind(textureUnit);
            bool useTexture = true;
            gpuProgram->setUniform(useTexture, "useTexture");

        }

        else {

            bool useTexture = false;
            gpuProgram->setUniform(useTexture, "useTexture");

        }

        gpuProgram->setUniform(material->kd, "kd");
        gpuProgram->setUniform(material->ks, "ks");
        gpuProgram->setUniform(material->ka, "ka");
        gpuProgram->setUniform(material->shine, "shine");

        gpuProgram->setUniform(state.light.La, "La");
        gpuProgram->setUniform(state.light.Le, "Le");
        gpuProgram->setUniform(state.light.wLiPos, "wLiPos");
        gpuProgram->setUniform(state.wEye, "wEye");

        for (unsigned int i = 0; i < 186; i++) {

            gpuProgram->setUniform(state.triangles[i], std::string("triangles[") + std::to_string(i) + std::string("]"));
      
       }

        geometry->DrawGeometry();

    }

};

void setTranslation(Object* obj, vec3 alapkozeppont, vec3 currentdir, vec3 dir, vec3 scale) {

    obj->scale = scale;

    vec3 axis = cross(currentdir, dir);
    float angle = acosf(dot(currentdir, dir));

    obj->rotationAxis = axis;
    obj->rotationAngle = angle;

    vec3 ndir = normalize(dir);

    vec3 translate = alapkozeppont;
    obj->translation = translate;

}

class Scene {

    std::vector<Object*> objects;
    Camera camera;
    Light light;
    vec4 triangles[186];

public:
    void Build() {

        vec3 kd1(0.1f, 0.2f, 0.3f);
        vec3 kd2(0.3f, 0.0f, 0.2f);
        vec3 kd3(0.3f, 0.2f, 0.1f);
        vec3 kd5(1.0f, 1.0f, 1.0f);
        vec3 ks(2.0f, 2.0f, 2.0f);
        vec3 ks2(0.0f, 0.0f, 0.0f);
        
        vec3 n(0.17f, 0.35f, 1.5f), kappa(3.1f, 2.7f, 1.9f);
        vec3 n2(1.3f, 1.3f, 1.3f), kappa2(0.0f, 0.0f, 0.0f);

        Material* sarga = new Material(kd3, ks, 3.0f * kd3, 50.0f);
        Material* cian = new Material(kd1, ks, 3.0f * kd1, 100.0f);
        Material* magenta = new Material(kd2, ks, 3.0f * kd2, 20.0f);

        Material* feher = new Material(kd5, ks2, 3.0f * kd5, 0.0f);

        Texture* txt = new CheckerBoardTexture(20, 20);

        Material* gold = new Material(n, kappa, 3.0f * n, 50);
        Material* water = new Material(n2, kappa2, 3.0f * n2, 50);

        Object3D* cylinder1 = new Cylinder();
        Object3D* cylinder2 = new Cylinder();
        Object3D* cylinder3 = new Cylinder();

        Object3D* cone1 = new Cone();
        Object3D* cone2 = new Cone();

        Object* cylinderobject1 = new Object(gold, cylinder1, nullptr);
        setTranslation(cylinderobject1, vec3(1, -1, 0), vec3(0, 0, -1), vec3(0.1, 1, 0), vec3(0.3, 0.3, 1));

        Object* cylinderobject2 = new Object(water, cylinder2, nullptr);
        setTranslation(cylinderobject2, vec3(0, -1, -0.8), vec3(0, 0, -1), vec3(-0.2, 1, 0.1), vec3(0.3, 0.3, 1));

        Object* cylinderobject3 = new Object(sarga, cylinder3, nullptr);
        setTranslation(cylinderobject3, vec3(-1, -1, 0), vec3(0, 0, -1), vec3(0, 1, -0.1), vec3(0.3, 0.3, 1));

        Object* coneobject1 = new Object(cian, cone1, nullptr);
        setTranslation(coneobject1, vec3(0, 1, 0), vec3(0, 0, 1), vec3(0.1, 1, -0.05), vec3(1, 1, 1));

        Object* coneobject2 = new Object(magenta, cone2, nullptr);
        setTranslation(coneobject2, vec3(0, 1, 0.8), vec3(0, 0, 1), vec3(-0.2, 1, 0), vec3(1, 1, 1));
        
        objects.push_back(cylinderobject1);
        objects.push_back(cylinderobject2);
        objects.push_back(cylinderobject3);

        objects.push_back(coneobject1);
        objects.push_back(coneobject2);

        Object3D* plane = new Plane();
        Object* planeobject = new Object(feher, plane,txt);
        planeobject->scale = vec3(20, 1, 20);
        objects.push_back(planeobject);

        camera.wEye = vec3(0.0f, 1.0f, 4.0f);
        camera.wLookat = vec3(0.0f, 0.0f, 0.0f);
        camera.wUp = vec3(0.0f, 1.0f, 0.0f);

        light.wLiPos = vec4(1.0f, 1.0f, 1.0f, 0.0f);
        light.La = vec3(0.4f, 0.4f, 0.4f);
        light.Le = vec3(2.0f, 2.0f, 2.0f);
    }

    void Render(GPUProgram* gpuProgram) {

        RenderState state;
        state.wEye = camera.wEye;
        state.V = camera.V();
        state.P = camera.P();
        state.light = light;

        int db = 0;
        int j = 0;
        for (Object* obj : objects) {

            mat4 M;
            obj->SetModelingTransformM(M);

            for (unsigned i = 0; i < obj->geometry->Vtx().size()-2;i++) {
     
                triangles[j] = M * vec4(obj->geometry->Vtx()[i].pos.x, obj->geometry->Vtx()[i].pos.y, obj->geometry->Vtx()[i].pos.z, 1);
                triangles[j+1] = M * vec4(obj->geometry->Vtx()[i+1].pos.x, obj->geometry->Vtx()[i+1].pos.y, obj->geometry->Vtx()[i+1].pos.z, 1);
                triangles[j+2] = M * vec4(obj->geometry->Vtx()[i+2].pos.x, obj->geometry->Vtx()[i+2].pos.y, obj->geometry->Vtx()[i+2].pos.z, 1);
                db += 3;
                j+=3;
                
            }
        }

        state.size = 62;
        for (int i = 0;i < 186;i++) {

            state.triangles[i] = vec3(triangles[i].x, triangles[i].y, triangles[i].z);
            
        }


        for (Object* obj : objects) obj->DrawObject(state, gpuProgram);

    }

    void Animate() { camera.Animate(); }

};

Scene scene;

class RaytraceApp : public glApp {

    GPUProgram* gpuProgram;

public:
    RaytraceApp() : glApp("Ray tracing") {}

    void onInitialization() {

        glViewport(0, 0, winWidth, winHeight);
        gpuProgram = new GPUProgram(vertSource, fragSource);

        glEnable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);

        scene.Build();

    }

    void onDisplay() {

        glClearColor(0.4, 0.4, 0.4, 1.0f);           
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
        scene.Render(gpuProgram);

    }

    void onKeyboard(int key) {

        if (key == 'a') {

            scene.Animate();
            refreshScreen();

        }
    }
}app;