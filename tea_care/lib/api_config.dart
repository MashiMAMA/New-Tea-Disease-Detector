// lib/api_config.dart
class ApiConfig {
  // IMPORTANT: Set this based on your device
  // true = Android Emulator
  // false = Physical Android Device
  static const bool isEmulator = true; // CHANGE THIS!

  // For Physical Device ONLY: Replace with your computer's IP
  // To find: Open CMD, type 'ipconfig', look for IPv4 Address
  static const String physicalDeviceIP = "192.168.1.100"; // CHANGE THIS!

  static const int port = 5000;

  static String get baseUrl {
    if (isEmulator) {
      return "http://10.0.2.2:$port";
    } else {
      return "http://$physicalDeviceIP:$port";
    }
  }

  static String get predictUrl => "$baseUrl/predict";
  static String get healthUrl => "$baseUrl/health";

  static const Duration timeout = Duration(seconds: 30);

  static void printConfig() {
    print("=" * 50);
    print("API Configuration");
    print("=" * 50);
    print("Device Type: ${isEmulator ? 'Emulator' : 'Physical Device'}");
    print("Base URL: $baseUrl");
    print("=" * 50);
  }
}
