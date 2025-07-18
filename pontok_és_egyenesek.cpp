
#include "framework.h"
#include <iostream>

//feladat megoldasa soran az oran elhangzottakat, valamint a grafika moodle oldalan fent levo power pointokat es pdf-eket hasznaltam
// csucspont arnyalo
const char * vertSource = R"(
	#version 330		

	layout(location = 0) in vec3 cP;	

	void main() {
		gl_Position = vec4(cP.x, cP.y, cP.z, 1); 	
	}
)";

// pixel arnyalo
const char * fragSource = R"(
	#version 330

	uniform vec3 color;			
	out vec4 fragmentColor;		

	void main() {
		fragmentColor = vec4(color, 1); 
	}
)";

//Az object osztaly
struct Object : public Geometry<vec3> {

	Object() {}
	//gyujtemeny megvaltoztatas
	void change(std::vector<vec3> vertices) {
		
		Vtx().clear();

		for (unsigned i = 0; i < vertices.size(); i++) {
			Vtx().push_back(vertices[i]);
		}
	}

};

struct PointCollection : public Geometry<vec3> {

	PointCollection(){}
	//uj pont hozzavetele
	void addPoint(vec3 p) {

		Vtx().push_back(p); 
		updateGPU();
		std::cout << "Point " << p.x << ", " << p.y << " added" << std::endl;

	}
	//pont kozelsegi merese
	vec3* near(vec3 point) {

		if (!Vtx().empty()) {

			vec3* nearest = &Vtx()[0];
			float mindistance = sqrt((Vtx()[0].x * Vtx()[0].x - 2 * Vtx()[0].x * point.x + point.x * point.x) + (Vtx()[0].y * Vtx()[0].y - 2 * Vtx()[0].y * point.y + point.y * point.y));

			for (unsigned i = 0; i < Vtx().size();i++) {

				float distance = sqrt((Vtx()[i].x * Vtx()[i].x - 2 * Vtx()[i].x * point.x + point.x * point.x) + (Vtx()[i].y * Vtx()[i].y - 2 * Vtx()[i].y * point.y + point.y * point.y));

				if (distance < mindistance) {

					nearest = &Vtx()[i];
					mindistance = distance;

				}

			}
			return nearest;
		}else {

			return nullptr;

		}
	}
	//pontok felrajzolasa kapott szinnel
	void DrawPoints(GPUProgram* gpuProgram, vec3 color) {

		Draw(gpuProgram, GL_POINTS, color);

	}
};

struct Line {
	float a;
	float b;
	float c;
	//kontrualas ket pontbol
	Line(vec3 p1, vec3 p2) {

		a = p2.y - p1.y;
		b = p1.x - p2.x;
		c = p2.x*p1.y - p1.x*p2.y;

		vec3 v = { p2.x - p1.x, p2.y - p1.y, p2.z - p1.z };

		std::cout << "Line added" << std::endl;
		std::cout << "Implicit:" << a << " x + " << b << " y + " << c <<  std::endl;
		std::cout << "Parametric: r(t) = (" << p1.x << ", " << p1.y << ") + (" << v.x << ", " << v.y << ")t" << std::endl;

	}
	//ket egyenes metszespontjanak meghatarozasa
	 vec3 intersect(Line* l) {

		vec3 intersect = { ((-1.0f * c * l->b) + (l->c * b)) / ((a * l->b) - (l->a * b)),((-1.0f * a * l->c) + (l->a * c))/ ((a * l->b) - (l->a * b)), 1.0f };
		return intersect;

	}
	 //rajta van-e egy pont az egyenesen
	 bool isonline(vec3 point) {
		 
		 float distance = abs(a * point.x + b * point.y + c) / sqrt(a*a + b*b);

		 if (distance < 0.01) {

			 return true;

		 }else {

			 return false;

		 }
	 }
	 //szakasz meghatarozas
	 Object* szakasz() {

		 std::vector<vec3> vec;
		 bool volt = false;

		 float a1, b1, c1;
		 a1 = 0;b1 = 1;c1 = 1;

		 //az ablak also egyenesenek implicit egyenlete 0x+1y+1=0
		 vec3 horizontaldown = { ((-1.0f * c * b1) + (c1 * b)) / ((a * b1) - (a1 * b)),((-1.0f * a * c1) + (a1 * c)) / ((a * b1) - (a1 * b)), 1.0f };

		 if (abs(horizontaldown.x) <= 1) {

			 vec.push_back(horizontaldown);

		 }

		 a1 = 0;b1 = 1;c1 = -1;

		 //az ablak felso egyenesenek implicit egyenlete 0x+1y-1=0
		 vec3 horizontalup = { ((-1.0f * c * b1) + (c1 * b)) / ((a * b1) - (a1 * b)),((-1.0f * a * c1) + (a1 * c)) / ((a * b1) - (a1 * b)), 1.0f };

		 if (abs(horizontalup.x) <= 1) {

			 volt = false;
			 for (unsigned i = 0;i < vec.size();i++) {

				 if (vec[i].x == horizontalup.x && vec[i].y == horizontalup.y) {

					 volt = true;

				 }
			 }

			 if (volt == false) {

				 vec.push_back(horizontalup);

			 }
			 
		 }

		 a1 = 1;b1 = 0;c1 = 1;

		 //az ablak bal oldali egyenesenek implicit egyenlete 1x+0y+1=0
		 vec3 verticalleft = { ((-1.0f * c * b1) + (c1 * b)) / ((a * b1) - (a1 * b)),((-1.0f * a * c1) + (a1 * c)) / ((a * b1) - (a1 * b)), 1.0f };

		 if (abs(verticalleft.y) <= 1) {

			 volt = false;
			 for (unsigned i = 0;i < vec.size();i++) {

				 if (vec[i].x == verticalleft.x && vec[i].y == verticalleft.y) {

					 volt = true;

				 }
			 }

			 if (volt == false) {

				 vec.push_back(verticalleft);

			 }
		 }

		 a1 = 1;b1 = 0;c1 = -1;

		 //az ablak jobb oldali egyenesenek implicit egyenlete 1x+0y-1=0
		 vec3 verticalright = { ((-1.0f * c * b1) + (c1 * b)) / ((a * b1) - (a1 * b)),((-1.0f * a * c1) + (a1 * c)) / ((a * b1) - (a1 * b)), 1.0f };

		 if (verticalright.y >= -1 && verticalright.y <= 1) {

			 volt = false;

			 for (unsigned i = 0;i < vec.size();i++) {

				 if (vec[i].x == verticalright.x && vec[i].y == verticalright.y) {

					 volt = true;

				 }
			 }

			 if (volt == false) {

				 vec.push_back(verticalright);

			 }
		 }

		 Object* egyenes = new Object();
		 egyenes->change(vec);
		 egyenes->updateGPU();

		 return egyenes;

	 }
	 //eltol egyenes
	 void eltol(vec3 point) {

		 c = -(a * point.x + b * point.y);

	 }
};


struct LineCollection {
	std::vector<Line*> lines ;
	
	LineCollection() {}
	//uj egyenes hozzaadasa
	void addLine(Line* l) {

		lines.push_back(l);

	}
	//pont koordinatai alapjan egy kozeli kivalasztasa
	Line* kozelebbi(vec3 point) {

		if (!lines.empty()) {

			Line* legkozelebb = lines[0];

			for (unsigned i = 1; i < lines.size();i++) {

				float distance1 = abs(legkozelebb->a * point.x + legkozelebb->b * point.y + legkozelebb->c) / sqrt(legkozelebb->a * legkozelebb->a + legkozelebb->b * legkozelebb->b);
				float distance2 = abs(lines[i]->a * point.x + lines[i]->b * point.y + lines[i]->c) / sqrt(lines[i]->a * lines[i]->a + lines[i]->b * lines[i]->b);

				if (distance2 < distance1) {

					legkozelebb = lines[i];

				}
			}

			return legkozelebb;
		}else {

			return nullptr;

		}
	}
	//egyenesek felrajzolasa a kapott szinnel
	void DrawLine(GPUProgram* gpuProgram, vec3 color) {

		for (unsigned i = 0;i < lines.size();i++) {
			Object* szakasz = lines[i]->szakasz();
			szakasz->Draw(gpuProgram, GL_LINES, color);

		}
	}

	~LineCollection() {
		for (unsigned i = 0;i < lines.size();i++) {
			delete[] lines[i];

		}
	}
};

enum State {
	POINT,    // Pont bevitel
	LINE,     // Egyenes felvétel
	MOVE,     // Egyenes mozgatás
	INTERSECT // Egyenesek metszése
};


const int winWidth = 600, winHeight = 600;

class MyApp : public glApp {
	State st;

	PointCollection* points = nullptr;
	LineCollection* ln = nullptr;

	vec3* selectedPoint1 = nullptr;
	vec3* selectedPoint2 = nullptr;
	Line* selectedLine1 = nullptr;
	Line* selectedLine2 = nullptr;
	Line* pickedLine = nullptr;

	GPUProgram gpuProgram;

public:
	MyApp() : glApp("Hazi") {

		st = POINT;

	}

	//inicializacio
	void onInitialization() {
		glPointSize(10);
		glLineWidth(3);
		points = new PointCollection();
		ln = new LineCollection();
		gpuProgram.create(vertSource, fragSource);
	}

	void onDisplay() {
		glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glViewport(0, 0, winWidth, winHeight);

		ln->DrawLine(&gpuProgram, vec3(0, 1, 1));
		points->DrawPoints(&gpuProgram, vec3(1,0,0));

		
	}

	//allapotok kivalasztasa
	void onKeyboard(int key) {
		
		switch (key) {
		case 'p': 

			st = POINT;
			std::cout << "Add point" << std::endl;

			break;
		case 'l': 

			st = LINE;
			std::cout << "Define lines" << std::endl;

			selectedPoint1 = nullptr;
			selectedPoint2 = nullptr;

			break;
		case 'm': 

			st = MOVE;
			std::cout << "Move" << std::endl;

			break;
		case 'i': 

			st = INTERSECT;
			std::cout << "Intersect" << std::endl;

			selectedLine1 = nullptr;
			selectedLine2 = nullptr;

			break;
		}
	}

	void onMousePressed(MouseButton button, int pX, int pY) {

		float cX = 2.0f * pX / winWidth - 1;	
		float cY = 1.0f - 2.0f * pY / winHeight;

		if (button == MOUSE_LEFT) {

			switch (st) {
			//pont lehelyezese
			case POINT:

				points->addPoint(vec3(cX, cY, 1.0f));

				break;
			//egyenes rajzolas
			case LINE:

				if (selectedPoint1 == nullptr) {

					selectedPoint1 = points->near(vec3(cX, cY, 1.0f));

				}else {

					selectedPoint2 = points->near(vec3(cX, cY, 1.0f));

					if (selectedPoint1 != selectedPoint2) {

						ln->addLine(new Line(*selectedPoint1, *selectedPoint2));

					}

					selectedPoint1 = nullptr;
					selectedPoint2 = nullptr;

				}

				break;
			//egyenes mozgatas
			case MOVE: {

				Line* l = ln->kozelebbi(vec3(cX, cY, 1.0f));

				if (l != nullptr) {

					if (l->isonline(vec3(cX, cY, 1.0f))) {
						pickedLine = l;

					}

				}

				break;
			}
			//ket kivalaszotott egyenes metszespontjara piros ponot tesz
			case INTERSECT:

				if (selectedLine1 == nullptr) {

					selectedLine1 = ln->kozelebbi(vec3(cX, cY, 1.0f));

				}else {

					selectedLine2 = ln->kozelebbi(vec3(cX, cY, 1.0f));

					if (selectedLine1 != selectedLine2) {

						vec3 point = selectedLine1->intersect(selectedLine2);

						if (abs(point.x) <= 1 && abs(point.y <= 1)) {

							points->addPoint(point);

						}
					}

					selectedLine1 = nullptr;
					selectedLine2 = nullptr;
				}
				break;
			}

		}
		refreshScreen();  
	}

	//egyenes mozgatas utan ennek elengedese
	void onMouseReleased(MouseButton button, int pX, int pY) {
		
		if (button == MOUSE_LEFT) {

			pickedLine = nullptr;
			
		}
	}

	void onMouseMotion(int pX, int pY) {
		float cX = 2.0f * pX / winWidth - 1;	
		float cY = 1.0f - 2.0f * pY / winHeight;

		if (st == MOVE) {
			if (pickedLine) {
				pickedLine->eltol(vec3(cX, cY, 1.0f));
			}
			refreshScreen();
		}
		

	}
}app;






