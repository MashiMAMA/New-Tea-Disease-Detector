import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:tea_care/image_load.dart';

class UploadImageScreen extends StatefulWidget {
  const UploadImageScreen({super.key});

  @override
  State<UploadImageScreen> createState() => _UploadImageScreenState();
}

class _UploadImageScreenState extends State<UploadImageScreen> {
  bool showOptions = false;
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage(ImageSource source) async {
    // On web, camera is not supported, only gallery
    if (kIsWeb && source == ImageSource.camera) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Camera not supported on web. Please select from files.'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    final pickedFile = await _picker.pickImage(source: source);
    if (pickedFile != null) {
      //navigation - pass XFile instead of File
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => ImageLoadScreen(imageFile: pickedFile),
        ),
      );

      print("Image selected: ${pickedFile.path}");
    } else {
      print("No image selected.");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Stack(
        children: [
          SafeArea(
            child: Column(
              children: [
                // Top green bar
                Container(
                  height: 50,
                  color: const Color(0xFF277B53),
                  alignment: Alignment.center,
                ),

                const SizedBox(height: 50),

                Center(
                  child: Transform.translate(
                    offset: const Offset(0, 40),
                    child: Image.asset(
                      'images/logo2.png',
                      height: 230,
                      width: 230,
                    ),
                  ),
                ),

                // Tagline
                const Text(
                  "Smarter Detection\nHealthier Leaves | Better Yields",
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.black54,
                  ),
                ),

                const SizedBox(height: 50),

                // Upload Button
                SizedBox(
                  width: 200,
                  height: 50,
                  child: ElevatedButton(
                    onPressed: () {
                      setState(() {
                        showOptions = !showOptions;
                      });
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF277B53),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                    child: const Text(
                      "Upload Image",
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w500,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ),

                const Spacer(),

                // Bottom image
                Expanded(
                  child: Transform.translate(
                    offset: const Offset(0, 0), // move up by 40px
                    child: Image.asset(
                      'images/tt2.png',
                      fit: BoxFit.cover,
                      width: double.infinity,
                    ),
                  ),
                ),
              ],
            ),
          ),

          // Sliding buttons
          AnimatedPositioned(
            duration: const Duration(milliseconds: 500),
            curve: Curves.easeOut,
            bottom:
                showOptions ? MediaQuery.of(context).size.height * 0.25 : -150,
            left: 32,
            right: 32,
            child: AnimatedOpacity(
              duration: const Duration(milliseconds: 500),
              opacity: showOptions ? 1.0 : 0.0,
              child: Column(
                children: [
                  ElevatedButton(
                    onPressed: () => _pickImage(ImageSource.camera),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color.fromARGB(255, 39, 123, 83),
                      foregroundColor: const Color.fromARGB(255, 254, 254, 254),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                      padding: const EdgeInsets.symmetric(
                        horizontal: 32,
                        vertical: 14,
                      ),
                    ),
                    child: const Text('Take a Photo'),
                  ),
                  const SizedBox(height: 12),
                  ElevatedButton(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color.fromARGB(255, 39, 123, 83),
                      foregroundColor: const Color.fromARGB(255, 255, 255, 255),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                      padding: const EdgeInsets.symmetric(
                        horizontal: 20,
                        vertical: 14,
                      ),
                    ),
                    child: const Text('Choose a Photo from Gallery'),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// import 'package:flutter/material.dart';
// import 'package:image_picker/image_picker.dart';
// import 'dart:io';
// import 'package:tea_care/image_load.dart';

// class UploadImageScreen extends StatefulWidget {
//   const UploadImageScreen({super.key});

//   @override
//   State<UploadImageScreen> createState() => _UploadImageScreenState();
// }

// class _UploadImageScreenState extends State<UploadImageScreen> {
//   bool showOptions = false;
//   final ImagePicker _picker = ImagePicker();

//   Future<void> _pickImage(ImageSource source) async {
//     final pickedFile = await _picker.pickImage(source: source);
//     if (pickedFile != null) {
//       File selectedImage = File(pickedFile.path);

//       Navigator.push(
//         context,
//         MaterialPageRoute(
//           builder: (context) => ImageLoadScreen(imageFile: selectedImage),
//         ),
//       );
//     } else {
//       debugPrint("No image selected.");
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     const greenColor = Color(0xFF2E7D32); // same green as your UI

//     return Scaffold(
//       backgroundColor: Colors.white,
//       body: Column(
//         children: [
//           // --- Top Green Bar ---
//           Container(
//             height: 50,
//             color: greenColor,
//           ),

//           // --- Main Content ---
//           Expanded(
//             child: Stack(
//               children: [
//                 Center(
//                   child: Padding(
//                     padding: const EdgeInsets.symmetric(horizontal: 32.0),
//                     child: Column(
//                       mainAxisAlignment: MainAxisAlignment.center,
//                       children: [
//                         // Logo
//                         Image.asset(
//                           'assets/images/Tcare.png',
//                           height: 120,
//                         ),
//                         const SizedBox(height: 20),

//                         // App Name
//                         const Text(
//                           'Tea Care',
//                           style: TextStyle(
//                             color: greenColor,
//                             fontSize: 32,
//                             fontWeight: FontWeight.bold,
//                           ),
//                         ),
//                         const SizedBox(height: 8),

//                         // Subtitle
//                         const Text(
//                           'Smarter Detection\nHealthier Leaves | Better Yields',
//                           textAlign: TextAlign.center,
//                           style: TextStyle(
//                             color: Colors.black87,
//                             fontSize: 16,
//                             height: 1.4,
//                           ),
//                         ),
//                         const SizedBox(height: 40),

//                         // Upload Image Button
//                         ElevatedButton(
//                           onPressed: () {
//                             setState(() {
//                               showOptions = !showOptions;
//                             });
//                           },
//                           style: ElevatedButton.styleFrom(
//                             backgroundColor: greenColor,
//                             foregroundColor: Colors.white,
//                             shape: RoundedRectangleBorder(
//                               borderRadius: BorderRadius.circular(30),
//                             ),
//                             padding: const EdgeInsets.symmetric(
//                               horizontal: 40,
//                               vertical: 16,
//                             ),
//                           ),
//                           child: const Text(
//                             'Upload Image',
//                             style: TextStyle(fontSize: 18),
//                           ),
//                         ),
//                       ],
//                     ),
//                   ),
//                 ),

//                 // --- Slide Up Buttons ---
//                 AnimatedPositioned(
//                   duration: const Duration(milliseconds: 500),
//                   curve: Curves.easeOut,
//                   bottom: showOptions ? 90 : -180,
//                   left: 32,
//                   right: 32,
//                   child: AnimatedOpacity(
//                     duration: const Duration(milliseconds: 500),
//                     opacity: showOptions ? 1.0 : 0.0,
//                     child: Column(
//                       children: [
//                         ElevatedButton(
//                           onPressed: () => _pickImage(ImageSource.camera),
//                           style: ElevatedButton.styleFrom(
//                             backgroundColor: Colors.white,
//                             foregroundColor: greenColor,
//                             shape: RoundedRectangleBorder(
//                               borderRadius: BorderRadius.circular(20),
//                               side: const BorderSide(color: greenColor),
//                             ),
//                             padding: const EdgeInsets.symmetric(
//                               horizontal: 32,
//                               vertical: 14,
//                             ),
//                           ),
//                           child: const Text('Take a Photo'),
//                         ),
//                         const SizedBox(height: 12),
//                         ElevatedButton(
//                           onPressed: () => _pickImage(ImageSource.gallery),
//                           style: ElevatedButton.styleFrom(
//                             backgroundColor: Colors.white,
//                             foregroundColor: greenColor,
//                             shape: RoundedRectangleBorder(
//                               borderRadius: BorderRadius.circular(20),
//                               side: const BorderSide(color: greenColor),
//                             ),
//                             padding: const EdgeInsets.symmetric(
//                               horizontal: 20,
//                               vertical: 14,
//                             ),
//                           ),
//                           child: const Text('Choose a Photo from Gallery'),
//                         ),
//                       ],
//                     ),
//                   ),
//                 ),
//               ],
//             ),
//           ),

//           // --- Bottom Green Bar ---
//           Container(
//             height: 100,
//             color: greenColor,
//           ),
//         ],
//       ),
//     );
//   }
// }

// import 'package:flutter/material.dart';
// import 'package:image_picker/image_picker.dart';
// import 'dart:io';
// import 'package:tea_care/image_load.dart';

// class UploadImageScreen extends StatefulWidget {
//   const UploadImageScreen({super.key});

//   @override
//   State<UploadImageScreen> createState() => _UploadImageScreenState();
// }

// class _UploadImageScreenState extends State<UploadImageScreen> {
//   bool showOptions = false;
//   final ImagePicker _picker = ImagePicker();

//   Future<void> _pickImage(ImageSource source) async {
//     final pickedFile = await _picker.pickImage(source: source);
//     if (pickedFile != null) {
//       File selectedImage = File(pickedFile.path);

//       Navigator.push(
//         context,
//         MaterialPageRoute(
//           builder: (context) => ImageLoadScreen(imageFile: selectedImage),
//         ),
//       );
//     } else {
//       debugPrint("No image selected.");
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       backgroundColor: Colors.white, // Clean white background
//       body: SafeArea(
//         child: Stack(
//           children: [
//             // --- Center Content ---
//             Center(
//               child: Padding(
//                 padding: const EdgeInsets.symmetric(horizontal: 32.0),
//                 child: Column(
//                   mainAxisAlignment: MainAxisAlignment.center,
//                   children: [
//                     // Logo
//                     Image.asset(
//                       'assets/images/Tcare.png',
//                       height: 120,
//                     ),
//                     const SizedBox(height: 20),

//                     // App Name
//                     const Text(
//                       'Tea Care',
//                       style: TextStyle(
//                         color: Colors.green,
//                         fontSize: 32,
//                         fontWeight: FontWeight.bold,
//                       ),
//                     ),
//                     const SizedBox(height: 8),

//                     // Subtitle
//                     const Text(
//                       'Smarter Detection\nHealthier Leaves | Better Yields',
//                       textAlign: TextAlign.center,
//                       style: TextStyle(
//                         color: Colors.black54,
//                         fontSize: 16,
//                         height: 1.4,
//                       ),
//                     ),
//                     const SizedBox(height: 40),

//                     // Upload Image Button
//                     ElevatedButton(
//                       onPressed: () {
//                         setState(() {
//                           showOptions = !showOptions;
//                         });
//                       },
//                       style: ElevatedButton.styleFrom(
//                         backgroundColor: Colors.green,
//                         foregroundColor: Colors.white,
//                         shape: RoundedRectangleBorder(
//                           borderRadius: BorderRadius.circular(30),
//                         ),
//                         padding: const EdgeInsets.symmetric(
//                           horizontal: 40,
//                           vertical: 16,
//                         ),
//                       ),
//                       child: const Text(
//                         'Upload Image',
//                         style: TextStyle(fontSize: 18),
//                       ),
//                     ),
//                   ],
//                 ),
//               ),
//             ),

//             // --- Slide Up Buttons ---
//             AnimatedPositioned(
//               duration: const Duration(milliseconds: 500),
//               curve: Curves.easeOut,
//               bottom: showOptions ? 100 : -180, // Slide up
//               left: 32,
//               right: 32,
//               child: AnimatedOpacity(
//                 duration: const Duration(milliseconds: 500),
//                 opacity: showOptions ? 1.0 : 0.0,
//                 child: Column(
//                   children: [
//                     ElevatedButton(
//                       onPressed: () => _pickImage(ImageSource.camera),
//                       style: ElevatedButton.styleFrom(
//                         backgroundColor: Colors.white,
//                         foregroundColor: Colors.green,
//                         shape: RoundedRectangleBorder(
//                           borderRadius: BorderRadius.circular(20),
//                           side: const BorderSide(color: Colors.green),
//                         ),
//                         padding: const EdgeInsets.symmetric(
//                           horizontal: 32,
//                           vertical: 14,
//                         ),
//                       ),
//                       child: const Text('Take a Photo'),
//                     ),
//                     const SizedBox(height: 12),
//                     ElevatedButton(
//                       onPressed: () => _pickImage(ImageSource.gallery),
//                       style: ElevatedButton.styleFrom(
//                         backgroundColor: Colors.white,
//                         foregroundColor: Colors.green,
//                         shape: RoundedRectangleBorder(
//                           borderRadius: BorderRadius.circular(20),
//                           side: const BorderSide(color: Colors.green),
//                         ),
//                         padding: const EdgeInsets.symmetric(
//                           horizontal: 20,
//                           vertical: 14,
//                         ),
//                       ),
//                       child: const Text('Choose a Photo from Gallery'),
//                     ),
//                   ],
//                 ),
//               ),
//             ),
//           ],
//         ),
//       ),
//     );
//   }
// }

// import 'package:flutter/material.dart';

// void main() {
//   runApp(const TeaCareApp());
// }

// class TeaCareApp extends StatelessWidget {
//   const TeaCareApp({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       debugShowCheckedModeBanner: false,
//       home: const OnboardingScreen(),
//     );
//   }
// }

// class OnboardingScreen extends StatelessWidget {
//   const OnboardingScreen({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       backgroundColor:
//           const Color.fromARGB(255, 102, 193, 106), // Green background
//       body: SafeArea(
//         child: Center(
//           child: Container(
//             margin: const EdgeInsets.all(20),
//             padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 24),
//             decoration: BoxDecoration(
//               color: Colors.white, // White card
//               borderRadius: BorderRadius.circular(24),
//             ),
//             child: Column(
//               mainAxisAlignment: MainAxisAlignment.center,
//               children: [
//                 // --- Image Section ---
//                 SizedBox(
//                   height: 250,
//                   child: Image.asset(
//                     "assets/images/Tcare.png", // <-- Replace with your tea image asset
//                     fit: BoxFit.contain,
//                   ),
//                 ),

//                 const SizedBox(height: 30),

//                 // --- App Title ---
//                 const Text(
//                   "Tea Care",
//                   style: TextStyle(
//                     fontSize: 28,
//                     fontWeight: FontWeight.bold,
//                     color: Colors.black87,
//                   ),
//                 ),

//                 const SizedBox(height: 12),

//                 // --- Subtitle ---
//                 const Text(
//                   "Smarter Detections,\nHealthier Leaves,\nBetter Yields",
//                   textAlign: TextAlign.center,
//                   style: TextStyle(
//                     fontSize: 16,
//                     color: Colors.black54,
//                     height: 1.5,
//                   ),
//                 ),

//                 const SizedBox(height: 40),

//                 // --- Next Button ---
//                 ElevatedButton(
//                   onPressed: () {
//                     // TODO: Navigate to next page
//                   },
//                   style: ElevatedButton.styleFrom(
//                     shape: const CircleBorder(),
//                     backgroundColor: const Color(0xFF93DA97), // Green button
//                     padding: const EdgeInsets.all(18),
//                     elevation: 3,
//                   ),
//                   child: const Icon(
//                     Icons.arrow_forward,
//                     color: Colors.white,
//                     size: 28,
//                   ),
//                 )
//               ],
//             ),
//           ),
//         ),
//       ),
//     );
//   }
// }

// import 'package:flutter/material.dart';
// import 'package:image_picker/image_picker.dart';
// import 'dart:io';
// import 'package:tea_care/image_load.dart';

// class UploadImageScreen extends StatefulWidget {
//   const UploadImageScreen({super.key});

//   @override
//   State<UploadImageScreen> createState() => _UploadImageScreenState();
// }

// class _UploadImageScreenState extends State<UploadImageScreen> {
//   bool showOptions = false;
//   final ImagePicker _picker = ImagePicker();

//   Future<void> _pickImage(ImageSource source) async {
//     final pickedFile = await _picker.pickImage(source: source);
//     if (pickedFile != null) {
//       File selectedImage = File(pickedFile.path);

//       Navigator.push(
//         context,
//         MaterialPageRoute(
//           builder: (context) => ImageLoadScreen(imageFile: selectedImage),
//         ),
//       );
//     } else {
//       print("No image selected.");
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       body: Stack(
//         fit: StackFit.expand,
//         children: [
//           Image.asset('images/background.jpg', fit: BoxFit.cover),
//           Container(color: Colors.black.withOpacity(0.4)),
//           Container(color: Colors.white.withOpacity(0.2)),
//           Center(
//             child: Padding(
//               padding: const EdgeInsets.symmetric(horizontal: 32.0),
//               child: Column(
//                 mainAxisAlignment: MainAxisAlignment.center,
//                 children: [
//                   const Text(
//                     'TeaCare',
//                     style: TextStyle(
//                       color: Colors.white,
//                       fontSize: 42,
//                       fontWeight: FontWeight.bold,
//                       fontFamily: 'Serif',
//                     ),
//                   ),
//                   const SizedBox(height: 16),
//                   const Text(
//                     'Smarter Detection,\nHealthier Leaves,\nBetter Yields.',
//                     textAlign: TextAlign.center,
//                     style: TextStyle(
//                       color: Colors.white70,
//                       fontSize: 16,
//                       height: 1.5,
//                     ),
//                   ),
//                   const SizedBox(height: 48),
//                   ElevatedButton(
//                     onPressed: () {
//                       setState(() {
//                         showOptions = !showOptions;
//                       });
//                     },
//                     style: ElevatedButton.styleFrom(
//                       backgroundColor: Colors.green.shade100,
//                       foregroundColor: Colors.black,
//                       shape: RoundedRectangleBorder(
//                         borderRadius: BorderRadius.circular(30),
//                       ),
//                       padding: const EdgeInsets.symmetric(
//                         horizontal: 40,
//                         vertical: 16,
//                       ),
//                     ),
//                     child: const Text('Upload Image',
//                         style: TextStyle(fontSize: 18)),
//                   ),
//                 ],
//               ),
//             ),
//           ),
//           AnimatedPositioned(
//             duration: const Duration(milliseconds: 500),
//             curve: Curves.easeOut,
//             bottom:
//                 showOptions ? MediaQuery.of(context).size.height * 0.2 : -150,
//             left: 32,
//             right: 32,
//             child: AnimatedOpacity(
//               duration: const Duration(milliseconds: 500),
//               opacity: showOptions ? 1.0 : 0.0,
//               child: Column(
//                 children: [
//                   ElevatedButton(
//                     onPressed: () => _pickImage(ImageSource.camera),
//                     style: ElevatedButton.styleFrom(
//                       backgroundColor: Colors.white,
//                       foregroundColor: Colors.black,
//                       shape: RoundedRectangleBorder(
//                         borderRadius: BorderRadius.circular(20),
//                       ),
//                       padding: const EdgeInsets.symmetric(
//                           horizontal: 32, vertical: 14),
//                     ),
//                     child: const Text('Take a Photo'),
//                   ),
//                   const SizedBox(height: 12),
//                   ElevatedButton(
//                     onPressed: () => _pickImage(ImageSource.gallery),
//                     style: ElevatedButton.styleFrom(
//                       backgroundColor: Colors.white,
//                       foregroundColor: Colors.black,
//                       shape: RoundedRectangleBorder(
//                         borderRadius: BorderRadius.circular(20),
//                       ),
//                       padding: const EdgeInsets.symmetric(
//                           horizontal: 20, vertical: 14),
//                     ),
//                     child: const Text('Choose a Photo from Gallery'),
//                   ),
//                 ],
//               ),
//             ),
//           ),
//         ],
//       ),
//     );
//   }
// }
