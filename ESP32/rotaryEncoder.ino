#include <ezButton.h>  // the library to use for SW pin

#define CLK_PIN 7       // ESP32 pin GPIO7 connected to the rotary encoder's CLK pin
#define DT_PIN  8       // ESP32 pin GPIO8 connected to the rotary encoder's DT pin
// #define SW_PIN  9       // ESP32 pin GPIO9 connected to the rotary encoder's SW pin
#define BTN_PIN  4

#define LED_PIN 8       // Built-in blue LED pin (ESP32 GPIO 8)

#define DIRECTION_CW  0   // clockwise direction
#define DIRECTION_CCW 1  // counter-clockwise direction

int direction = DIRECTION_CW;
int CLK_state;
int prev_CLK_state;
unsigned long lastInputTime = 0; // Tracks time of last valid input
const unsigned long debounceDelay = 100; // Minimum delay between inputs in milliseconds

ezButton button(BTN_PIN);  // create ezButton object that attach to BTN_PIN;

void setup() {
  Serial.begin(9600);

  // configure encoder pins as inputs
  pinMode(CLK_PIN, INPUT);
  pinMode(DT_PIN, INPUT);
  button.setDebounceTime(5);  // set debounce time to 5 milliseconds

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH); // Turn off LED (active LOW on most boards)

  // read the initial state of the rotary encoder's CLK pin
  prev_CLK_state = digitalRead(CLK_PIN);
}

void loop() {
  button.loop();  // MUST call the loop() function first

  // read the current state of the rotary encoder's CLK pin
  CLK_state = digitalRead(CLK_PIN);

  // If the state of CLK is changed and debounce time has passed
  if (CLK_state != prev_CLK_state && CLK_state == HIGH && (millis() - lastInputTime >= debounceDelay)) {
    // Fix: Swap the direction logic
    if (digitalRead(DT_PIN) == HIGH) {
      // The encoder is rotating in clockwise direction => increase the counter
      direction = DIRECTION_CW;
      Serial.println("+");
    } else {
      // The encoder is rotating in counter-clockwise direction => decrease the counter
      direction = DIRECTION_CCW;
      Serial.println("-");
    }

    lastInputTime = millis(); // Update the time of the last valid input

  }
  // save last CLK state
  prev_CLK_state = CLK_state;

  if (button.isPressed()) {
    Serial.println("btn");
  }
}
