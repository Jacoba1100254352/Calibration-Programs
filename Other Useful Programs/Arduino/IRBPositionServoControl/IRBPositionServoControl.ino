#include <SPI.h>
//#include <PID_v1.h>
#include <Servo.h>

/************************************************************************************************************************************************************
 *        Verify these are correct before each use
 * ********************************************************************************************************************************************************/

// calibration coefficients for each of the 4 sensors
//double coeff[4][2] = {{0.000520922, -2.292482252}, {0.000521521, -2.199142256}, {0.00052048, -2.429410973}, {0.000527817, -2.472489214}}; // Roger's original
double coeff[4][2] = { { 0.000520922, -2.083686515 }, { 0.000521521, -2.001596949 }, { 0.00052048, -2.232860712 }, { 0.000527817, -2.221581781 } };  //  Nick's
//double coeff[4][2] = {{0.00046542, -1.7638018}, {0.00049574, -2.1423678}, {0.00050083, -2.2047336}, {0.00050177, -1.9731005}}; // has blue wire, Daniel's

#define STARTING_SETPOINT 3  // (kPa) Can't exceed 10
//#define DIASTOLIC_PRESSURE 87       // (mmHg) Set according to measured pressure. Comment out if not using

#define PID_STARTING_PARAM false  // False to have locked position. True to have variable position.

//#define STARTING_CONTACT_AREA 497 // (mm^2) smaller platform
//#define STARTING_CONTACT_AREA 645 // (mm^2) bigger platform
//#define STARTING_CONTACT_AREA 631 // (mm^2) new platform with narrow clips
#define STARTING_CONTACT_AREA 144  // (mm^2) tappered platform for electrodes 2 x 8 mm, 4 mm edge to edge.
//#define STARTING_CONTACT_AREA 240 // (mm^2) tappered platform for electrodes 2 x 16 mm, 4 mm edge to edge.
//#define STARTING_CONTACT_AREA 32 // (mm^2) Pair of 2 x 8 mm electrodes.

/***********************************************************************************************************************************************************/

// -------------- Other Constants ---------------------

#define STARTING_SERVO_POSITION 0  // 0 is fully extended
#define TOLERANCE 0.2              // (kPa) can increase if too many oscilations.



#define ABSOLUTE_MAX_SETPOINT 10  // (kPa) This must stay 10 kPa or lower according to IRB
#define ABSOLUTE_MAX_PRESSURE 12  // (kPa) Alarm sounds if this pressure is exceeded at any time

#ifndef DIASTOLIC_PRESSURE
#define DIASTOLIC_PRESSURE 80
#endif

// ------------Configure Arduino and Servo-------------
Servo myservo;

const int chipSelectPins[] = { 7, 6, 5, 4 };  // SS
// MISO => pin 12
// SCLK => pin 13

const int servoPin = 9;

int currentPos = STARTING_SERVO_POSITION;  // 0- fully extended, 170 - least pressure
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
double contactArea = STARTING_CONTACT_AREA;  // Area of pressure plate that contacts skin mm^2
bool contactAreaReceiveMode = false;


// ------------Configure PID Controller-------------
double Input = 0;                     // Force (N)
double Output = 0;                    // Output (PWM)
double Setpoint = STARTING_SETPOINT;  // Pressure kPa
double MaxSetpoint = ABSOLUTE_MAX_SETPOINT;

//Specify the gains and initial tuning parameters
double Kp = 3;  // Originally was 3
//int Ki_original = 0;
//int Ki = Ki_original;
//int Kd = 10; // Originally was 5
boolean PIDenabled = PID_STARTING_PARAM;  // If true, the servo will move. If false, it won't.
double tolerance = TOLERANCE;             // kPa, can increase if too many oscilations. Originally 0.1

double TotalForce_N = 0.00;
double TotalPressure_kPa = 0.00;

// ---------- Configure Alarm ----------
#define alarmPin 19
#define BUZZER_MAX_POS_DELAY 200  // (ms)
bool toggleBuzzer = true;

//-------------Timing --------------
int waitTime = 200;  // (ms)
double enaStartTime = 0.0;

// ---------- PID and Buzzar Status _______
char* PID_Status = "ON";
char* BUZ_Status = "ON";


// 
void setup() {
  Serial.begin(9600);
  myservo.attach(servoPin);

  // 
  for (int i = 0; i < 4; i++) {
    pinMode(chipSelectPins[i], OUTPUT);
    digitalWrite(chipSelectPins[i], HIGH);
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

  // 
  myservo.write(currentPos);
}


void loop() {
  PID_Status = PIDenabled   ? "ON" : "OFF";
  BUZ_Status = toggleBuzzer ? "ON" : "OFF";

  // Check if there is a new input command on the serial port
  while (Serial.available() > 0)
    processInput();

  myservo.write(currentPos);
  delay(waitTime);

  readSensors();

  if (PIDenabled) {
    error = Setpoint - Input;

    if (abs(error) > tolerance) {
        commandOut = (int)(Kp * error);

        if (commandOut == 0)
            commandOut = (error < 0) ? -1 : 1;
    } else commandOut = 0;

    int newPos = currentPos - commandOut;
    if (newPos < minAngle)
      newPos = minAngle;
    else if (newPos > maxAngle)
      newPos = maxAngle;

    currentPos = newPos;
  }
  
  PrintToMonitor();
}


// 
void readSensors() {
  TotalForce_N = 0.00;
  TotalPressure_kPa = 0.00;

  for (int i = 0; i < 4; i++) {
    SPI.beginTransaction(SPISettings(800000, MSBFIRST, SPI_MODE0));
    digitalWrite(chipSelectPins[i], LOW);  // enable the selected sensor
    // First two bits are the status
    // Last 6 are first bits of force
    byte1in = SPI.transfer(0);
    // All 8 bits are force
    byte2in = SPI.transfer(0);
    digitalWrite(chipSelectPins[i], HIGH);  // disable the selected sensor
    SPI.endTransaction();

    byte1in_force = byte1in & B00111111;              // bitwise AND operation
    force_a2d[i] = (byte1in_force << 8) | (byte2in);  // bit shift operator and OR to combine all the bits of force (https://www.arduino.cc/reference/tr/language/structure/bitwise-operators/bitshiftleft/
    force_N[i] = coeff[i][0] * force_a2d[i] + coeff[i][1];
    if (force_N[i] < 0.00)
      force_N[i] = 0.00;
      
    TotalForce_N += force_N[i];
  }

  TotalPressure_kPa = TotalForce_N / contactArea * 1000;
  Input = TotalPressure_kPa;
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
    // New setpoint will be sent in the form of '<300>' which will be interpreted as 3 N
    // Divide by 100 is for if decimal value is sent (i.e. 350 => 3.5 N)
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

    case 'c':  // Sends contact area like "c500c"
        // Toggle contact area receive mode
        if (contactAreaReceiveMode = !contactAreaReceiveMode)
            contactArea = 0;

      break;

    case 'a':
      newPos = currentPos + commandOut;
      if (newPos > maxAngle)
        newPos = maxAngle;
      
      myservo.write(newPos);
      currentPos = newPos;
      break;
    
    
    case 'p':
      {
        double possibleSetpoint = 0.0;
        while (Serial.available()) {
          double nextdigit = Serial.read() - '0';
          if (nextdigit >= 0 && nextdigit <= 9)
            possibleSetpoint = possibleSetpoint * 10 + nextdigit;
        }
        
        if (Setpoint = possibleSetpoint > MaxSetpoint)
          PrintMaxTooHigh(Setpoint = MaxSetpoint);
      }
      break;

    case 's':
      {
        double possibleServoPos = 0.0;
        while (Serial.available()) {
          double nextdigit = Serial.read() - '0';
          if (nextdigit >= 0 && nextdigit <= 9)
            possibleServoPos = possibleServoPos * 10 + nextdigit;
        }

        if (possibleServoPos > maxAngle)
          possibleServoPos = maxAngle;
          
        newPos = possibleServoPos;
        currentPos = newPos;
      }
      break;

    default:
      //      Serial.print("In default state with char ");
      //      Serial.println((char)c);
      break;
  }
}


// 
bool triggerAlarm() {

  // Trigger alarm if band is fully extended or fully contracted
  bool hitLimit = (currentPos == minAngle || currentPos == maxAngle);

  if (hitLimit && (millis() - enaStartTime) > 5000 && PIDenabled) {
    enaStartTime = millis();
    Serial.println("     ******** Servo is fully extended *********      ");
    if (toggleBuzzer) {
      digitalWrite(alarmPin, HIGH);
      delay(BUZZER_MAX_POS_DELAY);
      digitalWrite(alarmPin, LOW);
      delay(BUZZER_MAX_POS_DELAY);
      digitalWrite(alarmPin, HIGH);
      delay(BUZZER_MAX_POS_DELAY);
      digitalWrite(alarmPin, LOW);
      delay(BUZZER_MAX_POS_DELAY);
      digitalWrite(alarmPin, HIGH);
      delay(BUZZER_MAX_POS_DELAY);
      digitalWrite(alarmPin, LOW);
      delay(BUZZER_MAX_POS_DELAY);
    }
  }

  // Trigger alarm if there is too much pressure
  if (TotalPressure_kPa > ABSOLUTE_MAX_PRESSURE) {
    if (toggleBuzzer)
      digitalWrite(alarmPin, HIGH);
    return true;
  } else {
    digitalWrite(alarmPin, LOW);
    return false;
  }
}


// 
void serialPrintF(const char *format, ...) {
  char buffer[128];  // Adjust buffer size if necessary
  va_list args;
  va_start(args, format);
  vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  Serial.print(buffer);
}


// 
void PrintToMonitor() {

  // Current Time
  serialPrintF("%lu\t", millis());
  
  // Status
  serialPrintF("PID %s, ", PID_Status);
  serialPrintF("Buz %s, ", BUZ_Status);

  // Position
  serialPrintF("Pos: %d\t\t", currentPos);

  // Force
  for (int i = 0; i < 4; i++)
    serialPrintF("%lu, ", force_a2d[i]);

  // Force
  for (int i = 0; i < 4; i++)
    Serial.print(String(force_N[i]) + ", ");

  // Total Force
  Serial.print(TotalForce_N);

  // Setpoint
  Serial.print("\tSetpoint: " + String(Setpoint) + ", ");

  // Total Pressure
  Serial.println("TotalPressure: " + String(Input));

  // Pressure Warning
  if (TotalPressure_kPa > ABSOLUTE_MAX_PRESSURE);
    //Serial.println("**********The total pressure is too high!!!********");
}


// Max Setpoint warning
void PrintMaxTooHigh(double MaxSetpoint) {
  serialPrintF("Can't set pressure above max pressure (Can't exceed %f kPa)", MaxSetpoint);
  delay(1000);
}

