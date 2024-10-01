#include "RTClib.h" // RTClib by Adafruit (v2.1.4 used in testing)
#include <SPI.h>    // Adafruit BusIO by Adafruit (v1.16.1 used in testing)

double coeff[4][2] = {{-0.00044280040320890596, 0.5050597471251378},
                      {-0.0005175249382435472, 1.9120768781888338},
                      {-0.0005141246242903676, 1.7785060726647088},
                      {-0.0005076996083303595,
                       1.6477909546137066}}; // New Sensor Set 1 (CM), Jacob's

const int chipSelectPins[] = {7, 6, 5, 4};
SPISettings HonewyWellFMASettings(800000, MSBFIRST, SPI_MODE0);
double contactArea = 144;
bool shouldReadSensors = true; // NOTE: Change to false for Serial Monitor
                               // output OR true for python->file output
byte byte1in, byte2in;
byte byte1in_force;
unsigned long force_a2d[4];
double force_N[4];
double TotalForce_N = 0.00;
double TotalPressure_kPa = 0.00;

unsigned long lastUpdate = 0;
const unsigned long interval = 20; // 20ms interval

// Boxcar averaging variables
const int boxcarSize = 10; // Number of samples for the moving average
double force_N_history[4][boxcarSize]; // History buffer for each sensor
int historyIndex = 0;
bool DISABLE_RTC = false;

// Setting up UNIX Epoch time
RTC_DS3231 rtc;

void setup() {
  Serial.begin(57600); // Lowering baud rate for stability, but you'll need to
                       // ensure the data fits within the time frame
  if (!rtc.begin() && !DISABLE_RTC) {
    Serial.println("Couldn't find RTC");
    while (1)
      ;
  }

  while (rtc.lostPower() && !DISABLE_RTC) {
    Serial.println("RTC lost power! The time needs to be reset!");
    // Serial.println("RTC lost power, let's set the time!");
    // // The following line sets the RTC to the date & time this sketch was
    // // compiled
    // rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
    // // This line sets the RTC with an explicit date & time, for example to set
    // // January 21, 2014 at 3am you would call:
    // // rtc.adjust(DateTime(2014, 1, 21, 3, 0, 0));
    // Serial.println("WARNING! Time will only be accurate to ±0.5 seconds.");
  }

  for (int i = 0; i < 4; i++) {
    pinMode(chipSelectPins[i], OUTPUT);
    digitalWrite(chipSelectPins[i], HIGH);

    // Initialize the history buffer to zero
    for (int j = 0; j < boxcarSize; j++) {
      force_N_history[i][j] = 0.0;
    }
  }

  SPI.begin();
}

void loop() {
  unsigned long currentMillis = millis();

  while (Serial.available() > 0)
    processInput();

  if (shouldReadSensors && currentMillis - lastUpdate >= interval &&
      (currentMillis - lastUpdate) % 20 == 0) {
    readSensors();
    PrintToMonitor();
    lastUpdate = currentMillis;
  }
}

void readSensors() {
  TotalForce_N = 0.00;
  TotalPressure_kPa = 0.00;

  for (int i = 0; i < 4; i++) {
    SPI.beginTransaction(SPISettings(800000, MSBFIRST, SPI_MODE0));
    digitalWrite(chipSelectPins[i], LOW);
    byte1in = SPI.transfer(0);
    byte2in = SPI.transfer(0);
    digitalWrite(chipSelectPins[i], HIGH);
    SPI.endTransaction();

    byte1in_force = byte1in & B00111111;
    force_a2d[i] = (byte1in_force << 8) | (byte2in);
    double raw_force = coeff[i][0] * force_a2d[i] + coeff[i][1];
    raw_force = -raw_force; // NOTE: Force has been calibrated to the negative
                            // force of the instron, so invert to make positive
    if (raw_force < 0.00)
      raw_force = 0.00;

    // Update the history buffer
    force_N_history[i][historyIndex] = raw_force;

    // Calculate the boxcar average
    double sum = 0.0;
    for (int j = 0; j < boxcarSize; j++) {
      sum += force_N_history[i][j];
    }
    force_N[i] = sum / boxcarSize;

    TotalForce_N += force_N[i];
  }
  TotalPressure_kPa = TotalForce_N / contactArea * 1000;

  // Update history index
  historyIndex = (historyIndex + 1) % boxcarSize;
}

void processInput() {
  byte c = Serial.read();
  if (c == 'r')
    shouldReadSensors = !shouldReadSensors;
}

void PrintToMonitor() {
  String output = String(millis()) + ",\t\t";
  for (int i = 0; i < 4; i++)
    output += String(force_a2d[i], DEC) + ", ";

  output += "\t";
  for (int i = 0; i < 4; i++)
    output += String(force_N[i], 4) +
              ", "; // Limit floating point precision // Changed from 2 to 4 to
                    // match instron precision

  output += "\t" + String(TotalForce_N, 2) + ", " +
            String(TotalPressure_kPa, 2) +
            ((!DISABLE_RTC) ? ", " : "");                   // Limit floating point precision
  if (!DISABLE_RTC)                                        // Append time if RTC is enabled
    output += String(rtc.now().unixtime()+21600); // Append UNIX Epoch time // VERIFY BEFORE RUNNING, 21,600 = 6*60*60 as it seems to be 6 hours behind somehow
  Serial.println(output);
}
