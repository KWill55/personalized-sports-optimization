/*
Title: sync_controller.ino

  Purpose: 
    - synchronize videos through an LED flash on ELEGOO microcontroller

  Prerequisites:
    - IR receiver module connected to the specified IR_PIN.
    - LED connected to the specified LED_PIN
    - IRremote library installed via the Arduino Library Manager.

  Usage:
    - Upload this sketch using the Arduino IDE.
    - Press any button on the IR remote to trigger a flash of the LED.
*/

// ================================
// Includes
// ================================

#include <IRremote.hpp>

// ================================
// Constants & Pin Definitions
// ================================

const int IR_PIN = 11;            // pin connected to IR receiver 
const int START_LED_PIN = 12;    // pin connected to start LEDs
const int STOP_LED_PIN = 10;     // pin connected to stop LEDs 

const int FLASH_DURATION = 15;   //Flash duration in milliseconds 

// ================================
// IR Receiver Setup
// ================================

IRrecv irrecv(IR_PIN);
decode_results results;

// ================================
// Setup
// ================================

void setup() {
  Serial.begin(9600);
  IrReceiver.begin(IR_PIN, ENABLE_LED_FEEDBACK);
  pinMode(START_LED_PIN, OUTPUT);
  pinMode(STOP_LED_PIN, OUTPUT);

}

// ================================
// Main Loop
// ================================

void loop() {
  if (IrReceiver.decode()) {
    unsigned long value = IrReceiver.decodedIRData.command;

    Serial.print("Received command: ");
    Serial.println(value, HEX);


    if (value == 0xC) { // corresponds to button 1 on remote 
      flashLED(START_LED_PIN);
    }

    if (value == 0x18) { //corresponds to button 2 on remote
      flashLED(STOP_LED_PIN);
    }



    IrReceiver.resume();  // Receive the next value
  }
}


// ================================
// Function: flashLED
// Description: Turns LED on for FLASH_DURATION ms
// ================================

void flashLED(int pin) {
  digitalWrite(pin, HIGH);
  delay(FLASH_DURATION);  
  digitalWrite(pin, LOW);
  delay(FLASH_DURATION);
}
