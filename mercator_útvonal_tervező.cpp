#include "framework.h"

const char* vertSource = R"(
	#version 330

	layout(location = 0) in vec2 vertexXY;	
	layout(location = 1) in vec2 vertexUV;			

	out vec2 texCoord;
							
	void main() {
		texCoord = vertexUV;										
		gl_Position = vec4(vertexXY, 0, 1); 		
	}
)";

const char* fragSource = R"(
	#version 330

	in vec2 texCoord;		
	out vec4 outColor;


	uniform sampler2D textureUnit;
	uniform vec3 color;
	uniform bool useTexture;	
	uniform float hour;	
	

	float tilt = 23.0f;
	const float pi = 3.14159265359;

	float hourdeg = hour*(pi/12.0f);
	

	float degtorad(float deg) {
		return deg * pi / 180.0f;
	}

	float minDegLat = -85.0f;
	float maxDegLat = 85.0f;
	float minDegLon = -180.0f;
	float maxDegLon = 180.0f;
	float minRadLon = degtorad(minDegLon);
	float maxRadLon = degtorad(maxDegLon);
	float ymax = log(tan(degtorad(maxDegLat / 2.0f + 45.0f))), ymin = log(tan(degtorad(minDegLat / 2.0f + 45.0f)));

	vec2 texturetondc(vec2 texture) {
		vec2 ndc;
		ndc.x = texture.x * 2.0f - 1.0f;
		ndc.y = texture.y * 2.0f - 1.0f;
		return ndc;
	}
	vec2 ndctomercator(vec2 ndc) {
		vec2 mercator;
		mercator.x = minRadLon + (maxRadLon - minRadLon) * (ndc.x + 1.0f) / 2.0f;
		mercator.y = ymin + (ymax - ymin) * (ndc.y + 1.0f) / 2.0f;
		return mercator;
	}
	vec2 mercatortogeographic(vec2 mercator) {
		vec2 geographic;
		geographic.x = mercator.x;
		geographic.y = 2.0f * atan(exp(mercator.y)) - (pi / 2.0f);
		return geographic;
	}

	float veclength(vec3 v) {
		return sqrt(v.x * v.x + v.y * v.y + v.z*v.z);
	}

	void main() { 
		if(useTexture){
		
			vec4 texColor = texture(textureUnit, texCoord); 

			vec2 sun = vec2(pi+hourdeg, degtorad(tilt));
			vec2 geographic = mercatortogeographic(ndctomercator(texturetondc(texCoord)));

			vec3 point1  = vec3(cos(sun.x)*cos(sun.y) , sin(sun.x)*cos(sun.y) , sin(sun.y) );
			vec3 point2 = vec3(cos(geographic.x)*cos(geographic.y) , sin(geographic.x)*cos(geographic.y) , sin(geographic.y) );

			float fi = dot(point1, point2)/(veclength(point1)*veclength(point2));
		
			if(fi<0.0f){
				texColor *= 0.5;
			}

			outColor = texColor; 
		}else{
			outColor = vec4(color,1);
		}
	}
)";

//globalis
const int winWidth = 600, winHeight = 600;

float degtorad(float deg) {
	return deg * M_PI / 180.0f;
}

const float minDegLat = -85, maxDegLat = 85;
const float minDegLon = -180, maxDegLon = 180;
const float minRadLat = degtorad(minDegLat), maxRadLat = degtorad(maxDegLat);
const float minRadLon = degtorad(minDegLon), maxRadLon = degtorad(maxDegLon);
const float ymax = log(tan(degtorad(maxDegLat / 2 + 45))), ymin = log(tan(degtorad(minDegLat / 2 + 45)));

vec3 geographictoworld(vec2 geographic) {
	return vec3(cos(geographic.x) * cos(geographic.y), sin(geographic.x) * cos(geographic.y), sin(geographic.y));
}

vec2 worldtogeographic(vec3 world) {
	vec2 geographic;
	geographic.x = atan2(world.y, world.x);
	geographic.y = asin(world.z);
	
	return geographic;
}

vec2 pixeltondc(vec2 pixel) {
	vec2 ndc;
	ndc.x = 2.0f * pixel.x / winWidth - 1;
	ndc.y = 1.0f - 2.0f * pixel.y / winHeight;
	return ndc;
}
vec2 ndctotexture(vec2 ndc) {
	vec2 texture;
	texture.x = (ndc.x + 1) / 2;
	texture.y = (ndc.y + 1) / 2;
	return texture;
}
vec2 texturetondc(vec2 texture) {
	vec2 ndc;
	ndc.x = texture.x * 2 - 1;
	ndc.y = texture.y * 2 - 1;
	return ndc;
}
vec2 ndctomercator(vec2 ndc) {
	vec2 mercator;
	mercator.x = minRadLon + (maxRadLon - minRadLon) * (ndc.x + 1) / 2;
	mercator.y = ymin + (ymax - ymin) * (ndc.y + 1) / 2;
	return mercator;
}
vec2 mercatortondc(vec2 mercator) {
	vec2 ndc;
	ndc.x = 2.0f * ((mercator.x - minRadLon) / (maxRadLon - minRadLon)) - 1;
	ndc.y = 2 * ((mercator.y - ymin) / (ymax - ymin)) - 1;
	return ndc;
}

vec2 mercatortogeographic(vec2 mercator) {
	vec2 geographic;
	geographic.x = mercator.x;
	geographic.y = 2.0f * atan(exp(mercator.y)) - M_PI / 2;
	return geographic;
}
vec2 geographictomercator(vec2 geographic) {
	vec2 mercator;
	mercator.x = geographic.x;
	mercator.y = log(tan(geographic.y / 2.0f + M_PI / 4));
	return mercator;
}

class txt {
	unsigned int textureId = 0;
public:
	txt(int width, int height, std::vector<vec3>&image) {
		glGenTextures(1, &textureId); // azonosító generálása
		glBindTexture(GL_TEXTURE_2D, textureId);    // ez az aktív innentõl
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}
	void Bind(int textureUnit) {
		glActiveTexture(GL_TEXTURE0 + textureUnit); // aktiválás
		glBindTexture(GL_TEXTURE_2D, textureId); // piros nyíl
	}
	~txt() {
		if (textureId > 0) glDeleteTextures(1, &textureId);
	}
};

class Terkep {

	unsigned int vao, vbo[2];	
	std::vector<vec2> vtx;	    
	txt* texture;         
	float hour;

	const vec3 palette[4] = {
	vec3(1.0f, 1.0f, 1.0f),  // feher
	vec3(0.0f, 0.0f, 1.0f),  // kek
	vec3(0.0f, 1.0f, 0.0f),  // zold
	vec3(0.0f, 0.0f, 0.0f)   // fekete
	};

public:

	Terkep() {

		std::vector<unsigned char> byteData = {
			252, 252, 252, 252, 252, 252, 252, 252, 252, 0, 9, 80, 1, 148, 13, 72, 13, 140, 25, 60, 21, 132, 41, 12, 1, 28,
			25, 128, 61, 0, 17, 4, 29, 124, 81, 8, 37, 116, 89, 0, 69, 16, 5, 48, 97, 0, 77, 0, 25, 8, 1, 8, 253, 253, 253, 253,
			101, 10, 237, 14, 237, 14, 241, 10, 141, 2, 93, 14, 121, 2, 5, 6, 93, 14, 49, 6, 57, 26, 89, 18, 41, 10, 57, 26,
			89, 18, 41, 14, 1, 2, 45, 26, 89, 26, 33, 18, 57, 14, 93, 26, 33, 18, 57, 10, 93, 18, 5, 2, 33, 18, 41, 2, 5, 2, 5, 6,
			89, 22, 29, 2, 1, 22, 37, 2, 1, 6, 1, 2, 97, 22, 29, 38, 45, 2, 97, 10, 1, 2, 37, 42, 17, 2, 13, 2, 5, 2, 89, 10, 49,
			46, 25, 10, 101, 2, 5, 6, 37, 50, 9, 30, 89, 10, 9, 2, 37, 50, 5, 38, 81, 26, 45, 22, 17, 54, 77, 30, 41, 22, 17, 58,
			1, 2, 61, 38, 65, 2, 9, 58, 69, 46, 37, 6, 1, 10, 9, 62, 65, 38, 5, 2, 33, 102, 57, 54, 33, 102, 57, 30, 1, 14, 33, 2,
			9, 86, 9, 2, 21, 6, 13, 26, 5, 6, 53, 94, 29, 26, 1, 22, 29, 0, 29, 98, 5, 14, 9, 46, 1, 2, 5, 6, 5, 2, 0, 13, 0, 13,
			118, 1, 2, 1, 42, 1, 4, 5, 6, 5, 2, 4, 33, 78, 1, 6, 1, 6, 1, 10, 5, 34, 1, 20, 2, 9, 2, 12, 25, 14, 5, 30, 1, 54, 13, 6,
			9, 2, 1, 32, 13, 8, 37, 2, 13, 2, 1, 70, 49, 28, 13, 16, 53, 2, 1, 46, 1, 2, 1, 2, 53, 28, 17, 16, 57, 14, 1, 18, 1, 14,
			1, 2, 57, 24, 13, 20, 57, 0, 2, 1, 2, 17, 0, 17, 2, 61, 0, 5, 16, 1, 28, 25, 0, 41, 2, 117, 56, 25, 0, 33, 2, 1, 2, 117,
			52, 201, 48, 77, 0, 121, 40, 1, 0, 205, 8, 1, 0, 1, 12, 213, 4, 13, 12, 253, 253, 253, 141
		};


		std::vector<vec3> decodedImage;

		for (unsigned char byte : byteData) {
			int H = byte >> 2;       
			int I = byte & 0b11;     
			for (int j = 0; j < H + 1; j++) {
				decodedImage.push_back(palette[I]);
			}
		}

		texture = new txt(64, 64, decodedImage);



		hour = 0.0f;

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(2, &vbo[0]);  // egyszerre ket vbo-t kerunk

		// a negyszog csucsai kezdetben normalizalt eszkozkoordinatakban
		vtx = { vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, 1) };
		
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, vtx.size() * sizeof(vec2), &vtx[0], GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0); 
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL); 

		// a negyszog csucsai texturaterben
		std::vector<vec2> uvs = { vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1) };
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); 
		glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(vec2), &uvs[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL); 

	}

	void hourplus() {
		if (hour < 24.0f) {

			hour += 1.0f;

		}
		else {

			hour = 0.0f;

		}
	}

	void Draw(GPUProgram* gpuProgram) {

		int textureUnit = 0; 
		gpuProgram->setUniform(textureUnit, "textureUnit"); 
		bool useTexture = true;
		gpuProgram->setUniform(useTexture, "useTexture");
		gpuProgram->setUniform(hour, "hour");
		texture->Bind(textureUnit);                        
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);  

	}

	virtual ~Terkep() {

		glDeleteBuffers(2, vbo);       
		glDeleteVertexArrays(1, &vao);

	}
};

class Ut : public Geometry<vec2> {
public:
	void createLine(vec2 p1, vec2 p2, bool last) {

		vec3 start = geographictoworld(mercatortogeographic(ndctomercator(p1)));
		vec3 end = geographictoworld(mercatortogeographic(ndctomercator(p2)));
		float theta = acos(dot(start, end));

		if(last) printf("Distance: %d km\n", (int)(6371 * theta));
		
		for (unsigned j = 0; j < 100; j++) {

			float t = (float)j / 100.0f;
			vec3 wVertex = sin((1 - t) * theta) / sin(theta) * start + sin(t * theta) / sin(theta) * end;
			Vtx().push_back(mercatortondc(geographictomercator(worldtogeographic(wVertex))));

		}

		updateGPU();

	}
	void DrawLine(GPUProgram* gpuProgram, vec3 color) {

		bool useTexture = false;
		gpuProgram->setUniform(useTexture, "useTexture");
		Draw(gpuProgram, GL_LINE_STRIP, color);

	}
};

class Allomas :public Geometry<vec2> {
public:
	void addPoint(vec2 p) {

		Vtx().push_back(p);
		updateGPU();

	}

	void DrawPoints(GPUProgram* gpuProgram, vec3 color) {

		for (unsigned i = 0; i < Vtx().size(); i++) {

			if ((i + 1) < Vtx().size()) {

				Ut u;

				if (i == (Vtx().size() - 2)) {

					u.createLine(Vtx()[i], Vtx()[i + 1], true);

				}else {

					u.createLine(Vtx()[i], Vtx()[i + 1], false);

				}

				u.DrawLine(gpuProgram, vec3(1, 1, 0));

			}
		}

		bool useTexture = false;
		gpuProgram->setUniform(useTexture, "useTexture");
		Draw(gpuProgram, GL_POINTS, color);

	}
};

class TextureApp : public glApp {

	GPUProgram* gpuProgram;
	Terkep* map;
	Allomas* allomas;

public:

	TextureApp() : glApp(3, 3, winWidth, winHeight, "Texturing") {}

	void onInitialization() {

		glPointSize(10);
		glLineWidth(3);

		map = new Terkep(); 
		allomas = new Allomas();
		gpuProgram = new GPUProgram(vertSource, fragSource);
		   

	}

	void onDisplay() {

		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);
		glViewport(0, 0, winWidth, winHeight);
		map->Draw(gpuProgram);      
		allomas->DrawPoints(gpuProgram, vec3(1,0,0));

	}

	void onMousePressed(MouseButton button, int pX, int pY) {
		vec2 ndc = pixeltondc(vec2(pX, pY));
	
		if (button == MOUSE_LEFT) {

			allomas->addPoint(vec2(ndc.x,ndc.y));
			refreshScreen();

		}	
	}

	void onKeyboard(int key) {

		if (key == 'n') {

			map->hourplus();
			refreshScreen();

		}
	}


};

TextureApp app;