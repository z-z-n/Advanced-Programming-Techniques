/*
Author: Zhining Zhang (903942246)
Class: ECE6122 (A)
Last Date Modified: 10/22/2923
Description:
Lab3: Use the keyboard to control the command.
*/
// Include GLFW
#include <glfw3.h>
extern GLFWwindow* window; // The "extern" keyword here is to access the variable "window" declared in tutorialXXX.cpp. This is a hack to keep the tutorials simple. Please avoid this.

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

#include "controls.hpp"

glm::mat4 ViewMatrix;
glm::mat4 ProjectionMatrix;
int off = 0; //0 = light up, -1 = off;

glm::mat4 getViewMatrix(){
	return ViewMatrix;
}
glm::mat4 getProjectionMatrix(){
	return ProjectionMatrix;
}
int getLightvalue() {
	return off;
}

// Initial position
glm::vec3 position = glm::vec3(0, 0, 0);
// Initial phi angle
float phi = 0.0f;
// Initial theta angle
float theta = 90.0f;
// units / s
float detheta = 50.0f;
// initial distance
float r = 10.0f;
// Initial Field of View
float initialFoV = 45.0f;

float speed = 3.0f; // 3 units / second



void computeMatricesFromInputs(){

	// glfwGetTime is called only once, the first time this function is called
	static double lastTime = glfwGetTime();

	// Compute time difference between current and last frame
	double currentTime = glfwGetTime();
	float deltaTime = float(currentTime - lastTime);

	// Move forward
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		r -= deltaTime * speed;
	}
	// Move backward
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		r += deltaTime * speed;
	}
	// Left rotate
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		phi += deltaTime * detheta;
	}
	// Right rotate
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		phi -= deltaTime * detheta;
	}
	// Up rotate
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
		theta -= deltaTime * detheta;
	}
	// Down rotate
	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
		theta += deltaTime * detheta;
	}
	// Close / turn on 
	if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) {
		off = ~off;
		//printf("Number = %d\n", off);
	}
	//check theta's value;
	if (theta < 0.0)
		theta = 0.1f;
	else if (theta > 180.0)
		theta = 179.8f;
	//calcultate positon;
	position.x = r * sin(glm::radians(theta)) * cos(glm::radians(phi));
	position.y = r * sin(glm::radians(theta)) * sin(glm::radians(phi));
	position.z = r * cos(glm::radians(theta));
	r = sqrt(position.x * position.x + position.y * position.y + position.z * position.z);
	float FoV = initialFoV;// - 5 * glfwGetMouseWheel(); // Now GLFW 3 requires setting up a callback for this. It's a bit too complicated for this beginner's tutorial, so it's disabled instead.

	// Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	ProjectionMatrix = glm::perspective(FoV, 4.0f / 3.0f, 0.1f, 100.0f);
	// Camera matrix
	ViewMatrix       = glm::lookAt(
								position,           // Camera is here
								glm::vec3(0.0f, 0.0f, 0.0f), // origion
								glm::vec3(0.0f, 0.0f, 1.0f)  // Head is up
						   );

	// For the next frame, the "last time" will be "now"
	lastTime = currentTime;
}