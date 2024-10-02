#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h> // for usleep

using namespace std;

#define GPIO_PATH "/sys/class/gpio/"

// Function to export GPIO
void exportGPIO(int pin) {
    ofstream exportFile(GPIO_PATH "export");
    if (!exportFile) {
        cerr << "Error: Unable to export GPIO pin" << endl;
        return;
    }
    exportFile << pin;
    exportFile.close();
}

// Function to unexport GPIO
void unexportGPIO(int pin) {
    ofstream unexportFile(GPIO_PATH "unexport");
    if (!unexportFile) {
        cerr << "Error: Unable to unexport GPIO pin" << endl;
        return;
    }
    unexportFile << pin;
    unexportFile.close();
}

// Function to set the GPIO direction
void setDirection(int pin, string direction) {
    ofstream directionFile(GPIO_PATH "gpio" + to_string(pin) + "/direction");
    if (!directionFile) {
        cerr << "Error: Unable to set direction for GPIO" << endl;
        return;
    }
    directionFile << direction;
    directionFile.close();
}

// Function to write GPIO value
void writeGPIO(int pin, int value) {
    ofstream valueFile(GPIO_PATH "gpio" + to_string(pin) + "/value");
    if (!valueFile) {
        cerr << "Error: Unable to write value to GPIO" << endl;
        return;
    }
    valueFile << value;
    valueFile.close();
}

// Function to read GPIO value
int readGPIO(int pin) {
    ifstream valueFile(GPIO_PATH "gpio" + to_string(pin) + "/value");
    int value;
    if (!valueFile) {
        cerr << "Error: Unable to read value from GPIO" << endl;
        return -1;
    }
    valueFile >> value;
    valueFile.close();
    return value;
}

int main() {
    int gpioPin = 216;  // Replace with the appropriate GPIO pin number

    // Export the GPIO pin
    exportGPIO(gpioPin);

    // Set the direction to output
    setDirection(gpioPin, "out");

    // Blink the LED connected to GPIO pin
    for (int i = 0; i < 10; i++) {
        writeGPIO(gpioPin, 1);  // Set GPIO pin high
        usleep(500000);         // Wait for 500 ms
        writeGPIO(gpioPin, 0);  // Set GPIO pin low
        usleep(500000);         // Wait for 500 ms
    }

    // Unexport the GPIO pin
    unexportGPIO(gpioPin);

    return 0;
}
