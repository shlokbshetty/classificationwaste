#include <Servo.h>
Servo wetServo, dryServo;

void setup() {
  Serial.begin(9600);
  wetServo.attach(8);
  dryServo.attach(9);
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'W') {
      Serial.println("Wet Bin Open");
      wetServo.write(90);  // open
      delay(2000);
      wetServo.write(0);   // close
    } else if (c == 'D') {
      Serial.println("Dry Bin Open");
      dryServo.write(90);
      delay(2000);
      dryServo.write(0);
    }
  }
}
