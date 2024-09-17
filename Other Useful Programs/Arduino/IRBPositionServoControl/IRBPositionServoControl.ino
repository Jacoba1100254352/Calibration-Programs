#include <SPI.h>
// #include <PID_v1.h>
#include <Servo.h>

/***********************************************************************************************************************************************************/
//        Verify these are correct before each use

// calibration coefficients for each of the 4 sensors
double coeff[4][2] = {
    {0.00044280040320890596, -0.5050597471251378},
    {0.0005175249382435472, -1.9120768781888338},
    {0.0005141246242903676, -1.7785060726647088},
    {0.0005076996083303595, -1.6477909546137066}}; // Calibrated

#define STARTING_SETPOINT 3.0 // (kPa) Can't exceed 10
#define DIASTOLIC_PRESSURE                                                     \
  78.0 // (mmHg) Set according to measured pressure. Comment out if not using

#define PID_STARTING_PARAM                                                     \
  true // False to have locked position. True to have variable position.

// Define platform area. (not electrode area)
// #define STARTING_CONTACT_AREA 144  // (mm^2) tapered platform for electrodes
// 2 x 8 mm, 4 mm edge to edge.
#define STARTING_CONTACT_AREA                                                  \
  240 // (mm^2) tapered platform for electrodes 2 x 16 mm, 4 mm edge to edge.

/***********************************************************************************************************************************************************/

// -------------- Other Constants ---------------------

#define STARTING_SERVO_POSITION 0 // 0 is fully extended
#define TOLERANCE 0.2 // (kPa) can increase if too many oscillations.

#define ABSOLUTE_MAX_SETPOINT                                                  \
  10 // (kPa) This must stay 10 kPa or lower according to IRB
#define ABSOLUTE_MAX_PRESSURE                                                  \
  12 // (kPa) Alarm sounds if this pressure is exceeded at any time

#define BOXCAR_ENABLED false // Set to false to disable boxcar averaging

#ifndef DIASTOLIC_PRESSURE
#define DIASTOLIC_PRESSURE 80
#endif

// ------------Configure Arduino and Servo-------------
Servo myservo;

const int chipSelectPins[] = {7, 6, 5, 4}; // SS
// MISO => pin 12
// SCLK => pin 13

const int servoPin = 9;

int commandOut = 0;
int minAngle = 0;
int maxAngle = 170;
double error = 0.00;

// ------------Configure HoneyWell force sensor -------------
SPISettings HonewyWellFMASettings(800000, MSBFIRST, SPI_MODE0);

// ------------Configure serial reader-------------
byte byte1in, byte2in;
byte byte1in_force;
unsigned long force_a2d[4];
double force_N[4];
const char startOfSerialDelimiter = '<';
const char endOfSerialDelimiter = '>';
int receivedNumber = 0;
double contactArea =
    STARTING_CONTACT_AREA; // Area of pressure plate that contacts skin mm^2
bool contactAreaReceiveMode = false;

// ------------Configure PID Controller-------------
double Input = 0;                    // Force (N)
double Output = 0;                   // Output (PWM)
double Setpoint = STARTING_SETPOINT; // Pressure kPa
double MaxSetpoint = ABSOLUTE_MAX_SETPOINT;

// Specify the gains and initial tuning parameters
double Kp = 3; // Originally was 3
// int Ki_original = 0;
// int Ki = Ki_original;
// int Kd = 10; // Originally was 5
bool PIDenabled =
    PID_STARTING_PARAM; // If true, the servo will move. If false, it won't.
double tolerance =
    TOLERANCE; // kPa, can increase if too many oscillations. Originally 0.1

double TotalForce_N = 0.00;
double TotalPressure_kPa = 0.00;

// ---------- Configure Alarm ----------
#define alarmPin 19
bool toggleBuzzer = true;

//-------------Timing --------------
int waitTime = 200; // (ms)
double enaStartTime = 0.0;
double moveStartTime = 0.0; // Time when the servo starts moving
bool isMoving = false;      // Flag to indicate if the servo is currently moving
const unsigned int SERVO_TIMEOUT =
    10000; // Time in ms before timout (10 seconds)

// ---------- PID and Buzzer Status ----------
char *PID_Status = "ON";
char *BUZ_Status = "ON";

// Boxcar averaging variables
const int boxcarSize = 10; // Number of samples for the moving average
double force_N_history[4][boxcarSize]; // History buffer for each sensor
int historyIndex = 0;

//
void setup() {
  Serial.begin(57600); // 9600 // 57600
  myservo.attach(servoPin);

  //
  for (int i = 0; i < 4; i++) {
    pinMode(chipSelectPins[i], OUTPUT);
    digitalWrite(chipSelectPins[i], HIGH);

    // Initialize the history buffer to zero
    for (int j = 0; j < boxcarSize; j++) {
      force_N_history[i][j] = 0.0;
    }
  }

  //
  double diastolicPressure_kPa = 0.133322 * DIASTOLIC_PRESSURE;
  Serial.println("\n\nDiastolic Pressure = " + String(diastolicPressure_kPa));

  // Verify Setpoint isn't too large
  if (diastolicPressure_kPa < ABSOLUTE_MAX_SETPOINT)
    MaxSetpoint = diastolicPressure_kPa;

  if (Setpoint > MaxSetpoint)
    Setpoint = MaxSetpoint;

  // Configure the alarm
  pinMode(alarmPin, OUTPUT);
  digitalWrite(alarmPin, LOW);

  // Start the SPI library:
  SPI.begin();

  // Don't move if the servo is told to go to an invalid position
  while (STARTING_SERVO_POSITION > 170 || STARTING_SERVO_POSITION < 0)
    Serial.println("Starting servo position must be between 0 and 170");

  // Move the servo to its starting point
  myservo.write(
      STARTING_SERVO_POSITION); // 0- fully extended, 170 - least pressure
}

void loop() {
  PID_Status = PIDenabled ? "ON" : "OFF";
  BUZ_Status = toggleBuzzer ? "ON" : "OFF";

  // Check if there is a new input command on the serial port
  while (Serial.available() > 0)
    processInput();

  delay(waitTime);

  readSensors(); // Read the sensor data

  if (PIDenabled) {
    error = Setpoint - Input;

    if (abs(error) > tolerance) {
      commandOut = (int)(Kp * error);

      if (commandOut == 0)
        commandOut = (error < 0) ? -1 : 1;
    } else
      commandOut = 0;

    // Determine the new position based on the error
    int newPos = myservo.read() - commandOut; // Compare the current position
    if (newPos < minAngle)
      newPos = minAngle;
    else if (newPos > maxAngle)
      newPos = maxAngle;

    myservo.write(newPos); // Write the new position to the servo
  }

  PrintToMonitor(); // Print relevant information to the serial monitor
}

//
void readSensors() {
  TotalForce_N = 0.00;
  TotalPressure_kPa = 0.00;

  for (int i = 0; i < 4; i++) {
    SPI.beginTransaction(SPISettings(800000, MSBFIRST, SPI_MODE0));
    digitalWrite(chipSelectPins[i], LOW); // enable the selected sensor
    // First two bits are the status
    // Last 6 are first bits of force
    byte1in = SPI.transfer(0);
    // All 8 bits are force
    byte2in = SPI.transfer(0);
    digitalWrite(chipSelectPins[i], HIGH); // disable the selected sensor
    SPI.endTransaction();

    byte1in_force = byte1in & B00111111; // bitwise AND operation
    force_a2d[i] =
        (byte1in_force << 8) |
        (byte2in); // bit shift operator and OR to combine all the bits of force
                   // (https://www.arduino.cc/reference/tr/language/structure/bitwise-operators/bitshiftleft/)
    double raw_force = coeff[i][0] * force_a2d[i] + coeff[i][1];
    if (raw_force < 0.00)
      raw_force = 0.00;

    // Update the history buffer
    force_N_history[i][historyIndex] = raw_force;

    // Boxcar Average
    if (BOXCAR_ENABLED) {
      // Calculate the boxcar average
      double sum = 0.0;
      for (int j = 0; j < boxcarSize; j++) {
        sum += force_N_history[i][j];
      }
      force_N[i] = sum / boxcarSize;
    } else {
      // If boxcar averaging is disabled, just use the raw force value
      force_N[i] = raw_force;
    }

    TotalForce_N += force_N[i];
  }

  TotalPressure_kPa = TotalForce_N / contactArea * 1000;
  Input = TotalPressure_kPa;

  // Update history index
  historyIndex = (historyIndex + 1) % boxcarSize;

  // Checks if alarm should be triggered and triggers it if applicable
  triggerAlarm();
}

//
void processInput() {
  byte c = Serial.read();
  int newPos = 0;

  // Debugging output to see which case is being executed
  //  Serial.print("Processing case: '");
  //  Serial.print((char)c);
  //  Serial.println("'");

  switch (c) {
  // New setpoint will be sent in the form of '<300>' which will be interpreted
  // as 3 N Divide by 100 is for if decimal value is sent (i.e. 350 => 3.5 N)
  case endOfSerialDelimiter:
    Setpoint = receivedNumber / 100.0;

    if (Setpoint > MaxSetpoint)
      PrintMaxTooHigh(Setpoint = MaxSetpoint);

    // Fall through to start a new number
  case startOfSerialDelimiter:
    receivedNumber = 0;
    break;

  case 'b':
    toggleBuzzer = !toggleBuzzer;
    break;

  case '0' ... '9':
    if (contactAreaReceiveMode == true) {
      contactArea *= 10;
      contactArea += c - '0';
    } else {
      receivedNumber *= 10;
      receivedNumber += c - '0';
    }
    break;

  case 'e':
    // Enable the PID controller
    PIDenabled = true;
    enaStartTime = millis();
    break;

  case 'd':
    // Disable the PID controller
    PIDenabled = false;
    break;

  case 'c': // Sends contact area like "c500c"
    // Toggle contact area receive mode
    if (contactAreaReceiveMode = !contactAreaReceiveMode)
      contactArea = 0;

    break;

  case 'a':
    newPos = myservo.read() + commandOut;
    if (newPos > maxAngle)
      newPos = maxAngle;

    myservo.write(newPos);
    break;

  case 'p': {
    double possibleSetpoint = 0.0;
    while (Serial.available()) {
      double nextdigit = Serial.read() - '0';
      if (nextdigit >= 0 && nextdigit <= 9)
        possibleSetpoint = possibleSetpoint * 10 + nextdigit;
    }

    if (Setpoint = possibleSetpoint > MaxSetpoint)
      PrintMaxTooHigh(Setpoint = MaxSetpoint);
  } break;

  default:
    //      Serial.print("In default state with char ");
    //      Serial.println((char)c);
    break;
  }
}

// Function to check if the alarm should be triggered and trigger it if
// necessary
bool triggerAlarm() {
  static int startTime = millis();
  static int previousPosition = myservo.read();

  // True if the servo has not moved for SERVO_TIMEOUT milliseconds
  const bool stallPeriodExceeded = (millis() - startTime) > SERVO_TIMEOUT;
  // True if the servo has not moved since the last check
  const bool potentialServoStall = previousPosition == myservo.read();
  // True if the servo has not moved and wants to move
  const bool servoWantsToMove = newPos != myservo.read();

  // Flag to determine if the alarm should be triggered
  bool alarmTriggered = false;

  // Check for potential servo stall
  if (stallPeriodExceeded && potentialServoStall && servoWantsToMove) {
    // Servo stall detected
    Serial.println(TotalPressure_kPa > ABSOLUTE_MAX_PRESSURE
                       ? "Pressure too high AND Servo appears stuck"
                       : "Servo appears stuck");

    if (toggleBuzzer) {
      digitalWrite(alarmPin, HIGH);
    }
    alarmTriggered = true; // Mark alarm as triggered
  }

  // Check if the pressure is too high
  if (TotalPressure_kPa > ABSOLUTE_MAX_PRESSURE && !alarmTriggered) {
    Serial.println("Pressure too high");

    if (toggleBuzzer) {
      digitalWrite(alarmPin, HIGH);
    }
    alarmTriggered = true; // Mark alarm as triggered
  }

  // If no alarm conditions are met, turn off the alarm
  if (!alarmTriggered) {
    digitalWrite(alarmPin, LOW);
  }

  // Update the previous position for the next check
  previousPosition = myservo.read();

  return alarmTriggered;
}

//
void PrintToMonitor() {
  // Current Time
  String output = String(millis()) + ",\t";

  // Status
  output += "PID " + String(PID_Status) + ", "; // PID Status
  output += "Buz " + String(BUZ_Status) + ", "; // Buzzer Status

  // Position
  output += "Pos: " + String(myservo.read()) + "\t";

  // Force
  output += "\tForce (ADC): ";
  for (int i = 0; i < 4; i++)
    output += String(force_a2d[i], DEC) + ", ";

  // Force
  output += "\tForce (N): ";
  for (int i = 0; i < 4; i++)
    output += String(force_N[i], 2) + ", "; // Limit floating point precision (2)

  // Total Force
  output += "\tTotalForce: " + String(TotalForce_N, 2) + "N, ";

  // Setpoint
  output += "\tSetpoint: " + String(Setpoint) + "(kPa), ";

  // Total Pressure (kPa)
  output += "TotalPressure: " + String(Input) + "(kPa), ";

  // Contact Area
  output += "\tContact Area: " + String(contactArea) + " mm^2";

  // Pressure Warning
  // if (TotalPressure_kPa > ABSOLUTE_MAX_PRESSURE)
  //   output += "\n**********The total pressure is too high!!!********";

  Serial.println(output);
}

// Max Setpoint warning
void PrintMaxTooHigh(double MaxSetpoint) {
  Serial.println("Can't set pressure above max pressure (Can't exceed " +
                 String(MaxSetpoint) + " kPa)");
  delay(1000);
}
