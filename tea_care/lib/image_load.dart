import 'dart:async';
import 'package:flutter/material.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:tea_care/api_config.dart';
import 'dart:typed_data';

class ImageLoadScreen extends StatefulWidget {
  final File imageFile;

  const ImageLoadScreen({super.key, required this.imageFile});

  @override
  State<ImageLoadScreen> createState() => _ImageLoadScreenState();
}

class _ImageLoadScreenState extends State<ImageLoadScreen> {
  bool _isDetected = false;
  bool _isLoading = false;
  String _diseaseName = "";
  String _scientificName = "";
  String _description = "";
  double _confidence = 0.0;
  String? _errorMessage;
  Map<String, double>? _allPredictions;
  Uint8List? _gradcamImage; // Store Grad-CAM heatmap image
  bool _showGradCAM = true; // Toggle between original and Grad-CAM

  @override
  void initState() {
    super.initState();
    ApiConfig.printConfig();
  }

  Future<void> _detectDisease() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _isDetected = false;
      _gradcamImage = null;
    });

    try {
      print("\n🔍 Starting disease detection with Grad-CAM...");
      print("📤 Sending to: ${ApiConfig.baseUrl}/predict_with_gradcam");

      var request = http.MultipartRequest(
        'POST',
        Uri.parse('${ApiConfig.baseUrl}/predict_with_gradcam'),
      );

      request.files.add(
        await http.MultipartFile.fromPath(
          'image',
          widget.imageFile.path,
        ),
      );

      print("📸 Image path: ${widget.imageFile.path}");

      var streamedResponse = await request.send().timeout(
        ApiConfig.timeout,
        onTimeout: () {
          throw Exception(
              'Connection timeout. Make sure Flask server is running.');
        },
      );

      var response = await http.Response.fromStream(streamedResponse);

      print("📥 Response status: ${response.statusCode}");

      if (response.statusCode == 200) {
        var jsonResponse = json.decode(response.body);
        print("✓ Response received successfully");

        if (jsonResponse['success'] == true) {
          setState(() {
            _isDetected = true;
            _diseaseName = jsonResponse['prediction']['disease_name'];
            _scientificName = jsonResponse['prediction']['scientific_name'];
            _description = jsonResponse['prediction']['description'];
            _confidence = jsonResponse['prediction']['confidence'].toDouble();
            _allPredictions =
                (jsonResponse['all_predictions'] as Map<String, dynamic>)
                    .map((key, value) => MapEntry(key, value.toDouble()));

            // Decode Grad-CAM image from base64
            if (jsonResponse['gradcam_image'] != null) {
              _gradcamImage = base64Decode(jsonResponse['gradcam_image']);
              print("✓ Grad-CAM image received");
            }
          });
          print("✓ Disease detected: $_diseaseName ($_confidence%)");
        } else {
          setState(() {
            _errorMessage =
                'Detection failed: ${jsonResponse['error'] ?? 'Unknown error'}';
          });
          print("✗ Detection failed: $_errorMessage");
        }
      } else {
        setState(() {
          _errorMessage =
              'Server error (${response.statusCode}): ${response.body}';
        });
        print("✗ Server error: $_errorMessage");
      }
    } on SocketException catch (e) {
      setState(() {
        _errorMessage = 'Network error: Cannot reach server.\n\n'
            'Make sure:\n'
            '• Flask server is running\n'
            '• You are using correct IP in api_config.dart\n'
            '• Phone and PC are on same WiFi';
      });
      print("✗ Socket error: $e");
    } on TimeoutException catch (e) {
      setState(() {
        _errorMessage = 'Connection timeout.\n\n'
            'Server took too long to respond.\n'
            'Check your connection and try again.';
      });
      print("✗ Timeout error: $e");
    } catch (e) {
      setState(() {
        _errorMessage = 'Error: ${e.toString()}';
      });
      print("✗ Unexpected error: $e");
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Widget _buildPredictionDetails() {
    if (_allPredictions == null || _allPredictions!.isEmpty) {
      return const SizedBox.shrink();
    }

    var sortedPredictions = _allPredictions!.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 16),
        const Text(
          "All Predictions",
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: Colors.black87,
          ),
        ),
        const SizedBox(height: 8),
        ...sortedPredictions.take(3).map((entry) {
          return Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    entry.key,
                    style: const TextStyle(fontSize: 13),
                  ),
                ),
                Text(
                  "${entry.value.toStringAsFixed(1)}%",
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: entry.key == _diseaseName
                        ? const Color(0xFF277B53)
                        : Colors.black54,
                  ),
                ),
              ],
            ),
          );
        }),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
        actions: [
          if (_gradcamImage != null)
            IconButton(
              icon: Icon(
                _showGradCAM ? Icons.image : Icons.crisis_alert,
                color: Colors.white,
              ),
              tooltip: _showGradCAM ? 'Show Original' : 'Show Grad-CAM',
              onPressed: () {
                setState(() {
                  _showGradCAM = !_showGradCAM;
                });
              },
            ),
        ],
      ),
      body: Column(
        children: [
          Container(
            height: 80,
            color: const Color(0xFF277B53),
            alignment: Alignment.center,
          ),
          // Image display with Grad-CAM toggle
          ClipRRect(
            borderRadius: const BorderRadius.only(
              bottomLeft: Radius.circular(24),
              bottomRight: Radius.circular(24),
            ),
            child: Stack(
              children: [
                // Original or Grad-CAM image
                if (_showGradCAM && _gradcamImage != null)
                  Image.memory(
                    _gradcamImage!,
                    height: MediaQuery.of(context).size.height * 0.45,
                    width: double.infinity,
                    fit: BoxFit.cover,
                  )
                else
                  Image.file(
                    widget.imageFile,
                    height: MediaQuery.of(context).size.height * 0.45,
                    width: double.infinity,
                    fit: BoxFit.cover,
                  ),

                // Label overlay
                if (_gradcamImage != null)
                  Positioned(
                    bottom: 8,
                    left: 8,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.7),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            _showGradCAM ? Icons.crisis_alert : Icons.image,
                            color: Colors.white,
                            size: 16,
                          ),
                          const SizedBox(width: 6),
                          Text(
                            _showGradCAM ? 'Grad-CAM View' : 'Original Image',
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          ),
          // Results container
          Expanded(
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: const BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.only(
                  topLeft: Radius.circular(24),
                  topRight: Radius.circular(24),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (_isLoading)
                    const Expanded(
                      child: Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            CircularProgressIndicator(
                              color: Color(0xFF277B53),
                              strokeWidth: 3,
                            ),
                            SizedBox(height: 16),
                            Text(
                              "Analyzing leaf image...",
                              style: TextStyle(
                                fontSize: 16,
                                color: Colors.black54,
                              ),
                            ),
                            SizedBox(height: 8),
                            Text(
                              "Generating Grad-CAM visualization",
                              style: TextStyle(
                                fontSize: 12,
                                color: Colors.black38,
                              ),
                            ),
                          ],
                        ),
                      ),
                    )
                  else if (_errorMessage != null) ...[
                    Expanded(
                      child: SingleChildScrollView(
                        child: Container(
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: Colors.red.shade50,
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(color: Colors.red.shade200),
                          ),
                          child: Column(
                            children: [
                              const Icon(
                                Icons.error_outline,
                                color: Colors.red,
                                size: 48,
                              ),
                              const SizedBox(height: 12),
                              const Text(
                                "Detection Failed",
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.red,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                _errorMessage!,
                                textAlign: TextAlign.center,
                                style: const TextStyle(
                                  fontSize: 14,
                                  color: Colors.black87,
                                  height: 1.5,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ] else if (_isDetected) ...[
                    Expanded(
                      child: SingleChildScrollView(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            // Disease name with icon
                            Row(
                              children: [
                                Container(
                                  padding: const EdgeInsets.all(8),
                                  decoration: BoxDecoration(
                                    color: const Color(0xFF277B53)
                                        .withOpacity(0.1),
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Icon(
                                    _diseaseName == "Healthy Leaf"
                                        ? Icons.check_circle
                                        : Icons.warning_amber_rounded,
                                    color: _diseaseName == "Healthy Leaf"
                                        ? Colors.green
                                        : Colors.orange,
                                    size: 28,
                                  ),
                                ),
                                const SizedBox(width: 12),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        _diseaseName,
                                        style: const TextStyle(
                                          fontSize: 22,
                                          fontWeight: FontWeight.bold,
                                          color: Colors.black87,
                                        ),
                                      ),
                                      const SizedBox(height: 4),
                                      Text(
                                        _scientificName,
                                        style: const TextStyle(
                                          fontSize: 14,
                                          fontStyle: FontStyle.italic,
                                          color: Colors.black54,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 12),
                            // Confidence badge
                            Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 16,
                                vertical: 8,
                              ),
                              decoration: BoxDecoration(
                                color: const Color(0xFF277B53).withOpacity(0.1),
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  const Icon(
                                    Icons.analytics_outlined,
                                    size: 16,
                                    color: Color(0xFF277B53),
                                  ),
                                  const SizedBox(width: 6),
                                  Text(
                                    "Confidence: ${_confidence.toStringAsFixed(1)}%",
                                    style: const TextStyle(
                                      fontSize: 14,
                                      fontWeight: FontWeight.w600,
                                      color: Color(0xFF277B53),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            const SizedBox(height: 20),
                            // Description
                            const Text(
                              "Description & Treatment",
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: Colors.black87,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              _description,
                              style: const TextStyle(
                                fontSize: 14,
                                color: Colors.black87,
                                height: 1.6,
                              ),
                            ),
                            // All predictions
                            _buildPredictionDetails(),
                          ],
                        ),
                      ),
                    ),
                  ] else ...[
                    const Expanded(
                      child: Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.cloud_upload_outlined,
                              size: 64,
                              color: Colors.black26,
                            ),
                            SizedBox(height: 16),
                            Text(
                              "Ready to Detect",
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.w600,
                                color: Colors.black54,
                              ),
                            ),
                            SizedBox(height: 8),
                            Text(
                              "Press 'Detect Disease' to analyze this leaf",
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                fontSize: 14,
                                color: Colors.black38,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                  const SizedBox(height: 16),
                  // Detect button
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: _isLoading ? null : _detectDisease,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF277B53),
                        disabledBackgroundColor: Colors.grey,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        elevation: 2,
                      ),
                      child: Text(
                        _isDetected ? "Detect Again" : "Detect Disease",
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 10),
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
// import 'dart:io';
// import 'package:http/http.dart' as http;
// import 'dart:convert';

// class ImageLoadScreen extends StatefulWidget {
//   final File imageFile;

//   const ImageLoadScreen({super.key, required this.imageFile});

//   @override
//   State<ImageLoadScreen> createState() => _ImageLoadScreenState();
// }

// class _ImageLoadScreenState extends State<ImageLoadScreen> {
//   bool _isDetected = false;
//   bool _isLoading = false;
//   String _diseaseName = "";
//   String _scientificName = "";
//   String _description = "";
//   double _confidence = 0.0;
//   String? _errorMessage;

//   // IMPORTANT: Change this to your local IP address
//   // For Android Emulator: use "10.0.2.2"
//   // For physical device: use your computer's local IP (e.g., "192.168.1.100")
//   final String baseUrl = "http://10.0.2.2:5000"; // Change this!

//   Future<void> _detectDisease() async {
//     setState(() {
//       _isLoading = true;
//       _errorMessage = null;
//     });

//     try {
//       // Create multipart request
//       var request = http.MultipartRequest(
//         'POST',
//         Uri.parse('$baseUrl/predict'),
//       );

//       // Add image file
//       request.files.add(
//         await http.MultipartFile.fromPath(
//           'image',
//           widget.imageFile.path,
//         ),
//       );

//       // Send request
//       var streamedResponse = await request.send();
//       var response = await http.Response.fromStream(streamedResponse);

//       if (response.statusCode == 200) {
//         // Parse JSON response
//         var jsonResponse = json.decode(response.body);

//         if (jsonResponse['success'] == true) {
//           setState(() {
//             _isDetected = true;
//             _diseaseName = jsonResponse['prediction']['disease_name'];
//             _scientificName = jsonResponse['prediction']['scientific_name'];
//             _description = jsonResponse['prediction']['description'];
//             _confidence = jsonResponse['prediction']['confidence'];
//           });
//         } else {
//           setState(() {
//             _errorMessage = 'Detection failed: ${jsonResponse['error']}';
//           });
//         }
//       } else {
//         setState(() {
//           _errorMessage = 'Server error: ${response.statusCode}';
//         });
//       }
//     } catch (e) {
//       setState(() {
//         _errorMessage = 'Connection error: ${e.toString()}';
//       });
//       print('Error: $e');
//     } finally {
//       setState(() {
//         _isLoading = false;
//       });
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       extendBodyBehindAppBar: true,
//       appBar: AppBar(
//         backgroundColor: Colors.transparent,
//         elevation: 0,
//         leading: IconButton(
//           icon: const Icon(Icons.arrow_back, color: Colors.white),
//           onPressed: () {
//             Navigator.pop(context);
//           },
//         ),
//       ),
//       body: Column(
//         children: [
//           // Top green bar
//           Container(
//             height: 80,
//             color: const Color(0xFF277B53),
//             alignment: Alignment.center,
//           ),
//           // Top image
//           ClipRRect(
//             borderRadius: const BorderRadius.only(
//               bottomLeft: Radius.circular(24),
//               bottomRight: Radius.circular(24),
//             ),
//             child: Image.file(
//               widget.imageFile,
//               height: MediaQuery.of(context).size.height * 0.45,
//               width: double.infinity,
//               fit: BoxFit.cover,
//             ),
//           ),
//           // White container with rounded corners
//           Expanded(
//             child: Container(
//               width: double.infinity,
//               padding: const EdgeInsets.all(20),
//               decoration: const BoxDecoration(
//                 color: Colors.white,
//                 borderRadius: BorderRadius.only(
//                   topLeft: Radius.circular(24),
//                   topRight: Radius.circular(24),
//                 ),
//               ),
//               child: Column(
//                 crossAxisAlignment: CrossAxisAlignment.start,
//                 children: [
//                   if (_isLoading)
//                     const Center(
//                       child: Column(
//                         children: [
//                           CircularProgressIndicator(
//                             color: Color(0xFF277B53),
//                           ),
//                           SizedBox(height: 16),
//                           Text(
//                             "Analyzing image...",
//                             style: TextStyle(
//                               fontSize: 16,
//                               color: Colors.black54,
//                             ),
//                           ),
//                         ],
//                       ),
//                     )
//                   else if (_errorMessage != null)
//                     Container(
//                       padding: const EdgeInsets.all(12),
//                       decoration: BoxDecoration(
//                         color: Colors.red.shade50,
//                         borderRadius: BorderRadius.circular(8),
//                       ),
//                       child: Row(
//                         children: [
//                           const Icon(Icons.error_outline, color: Colors.red),
//                           const SizedBox(width: 8),
//                           Expanded(
//                             child: Text(
//                               _errorMessage!,
//                               style: const TextStyle(color: Colors.red),
//                             ),
//                           ),
//                         ],
//                       ),
//                     )
//                   else if (_isDetected) ...[
//                     // Disease name
//                     Text(
//                       _diseaseName,
//                       style: const TextStyle(
//                         fontSize: 22,
//                         fontWeight: FontWeight.bold,
//                         color: Colors.black87,
//                       ),
//                     ),
//                     const SizedBox(height: 8),
//                     // Scientific name
//                     Text(
//                       _scientificName,
//                       style: const TextStyle(
//                         fontSize: 16,
//                         fontStyle: FontStyle.italic,
//                         color: Colors.black54,
//                       ),
//                     ),
//                     const SizedBox(height: 8),
//                     // Confidence
//                     Container(
//                       padding: const EdgeInsets.symmetric(
//                         horizontal: 12,
//                         vertical: 6,
//                       ),
//                       decoration: BoxDecoration(
//                         color: const Color(0xFF277B53).withOpacity(0.1),
//                         borderRadius: BorderRadius.circular(20),
//                       ),
//                       child: Text(
//                         "Confidence: ${_confidence.toStringAsFixed(1)}%",
//                         style: const TextStyle(
//                           fontSize: 14,
//                           fontWeight: FontWeight.w600,
//                           color: Color(0xFF277B53),
//                         ),
//                       ),
//                     ),
//                     const SizedBox(height: 16),
//                     // Description header
//                     const Text(
//                       "Description",
//                       style: TextStyle(
//                         fontSize: 18,
//                         fontWeight: FontWeight.bold,
//                         color: Colors.black,
//                       ),
//                     ),
//                     const SizedBox(height: 8),
//                     // Description text
//                     Expanded(
//                       child: SingleChildScrollView(
//                         child: Text(
//                           _description,
//                           style: const TextStyle(
//                             fontSize: 14,
//                             color: Colors.black87,
//                             height: 1.5,
//                           ),
//                         ),
//                       ),
//                     ),
//                   ] else ...[
//                     const Center(
//                       child: Column(
//                         mainAxisAlignment: MainAxisAlignment.center,
//                         children: [
//                           Icon(
//                             Icons.info_outline,
//                             size: 48,
//                             color: Colors.black26,
//                           ),
//                           SizedBox(height: 16),
//                           Text(
//                             "Press 'Detect Disease' to analyze the leaf",
//                             textAlign: TextAlign.center,
//                             style: TextStyle(
//                               fontSize: 16,
//                               color: Colors.black54,
//                             ),
//                           ),
//                         ],
//                       ),
//                     ),
//                   ],
//                   const SizedBox(height: 16),
//                   // Buttons
//                   Column(
//                     children: [
//                       SizedBox(
//                         width: double.infinity,
//                         child: ElevatedButton(
//                           onPressed: _isLoading ? null : _detectDisease,
//                           style: ElevatedButton.styleFrom(
//                             backgroundColor: const Color(0xFF277B53),
//                             padding: const EdgeInsets.symmetric(vertical: 16),
//                             shape: RoundedRectangleBorder(
//                               borderRadius: BorderRadius.circular(12),
//                             ),
//                           ),
//                           child: Text(
//                             _isDetected ? "Detect Again" : "Detect Disease",
//                             style: const TextStyle(
//                               fontSize: 16,
//                               color: Colors.white,
//                             ),
//                           ),
//                         ),
//                       ),
//                       const SizedBox(height: 10),
//                     ],
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


































// import 'package:flutter/material.dart';
// import 'dart:io';

// class ImageLoadScreen extends StatefulWidget {
//   final File imageFile;

//   const ImageLoadScreen({super.key, required this.imageFile});

//   @override
//   State<ImageLoadScreen> createState() => _ImageLoadScreenState();
// }

// class _ImageLoadScreenState extends State<ImageLoadScreen> {
//   bool _isDetected = false;

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       extendBodyBehindAppBar: true,
//       appBar: AppBar(
//         backgroundColor: Colors.transparent,
//         elevation: 0,
//         leading: IconButton(
//           icon: const Icon(Icons.arrow_back, color: Colors.white),
//           onPressed: () {
//             Navigator.pop(context);
//           },
//         ),
//       ),
//       body: Column(
//         children: [
//           // Top green bar
//           Container(
//             height: 80,
//             color: const Color(0xFF277B53),
//             alignment: Alignment.center,
//           ),
//           // Top image
//           ClipRRect(
//             borderRadius: const BorderRadius.only(
//               bottomLeft: Radius.circular(24),
//               bottomRight: Radius.circular(24),
//             ),
//             child: Image.file(
//               widget.imageFile,
//               height: MediaQuery.of(context).size.height * 0.45,
//               width: double.infinity,
//               fit: BoxFit.cover,
//             ),
//           ),
//           // White container with rounded corners
//           Expanded(
//             child: Container(
//               width: double.infinity,
//               padding: const EdgeInsets.all(20),
//               decoration: const BoxDecoration(
//                 color: Colors.white,
//                 borderRadius: BorderRadius.only(
//                   topLeft: Radius.circular(24),
//                   topRight: Radius.circular(24),
//                 ),
//               ),
//               child: Column(
//                 crossAxisAlignment: CrossAxisAlignment.start,
//                 children: [
//                   if (_isDetected) ...[
//                     // Disease name placeholder
//                     const Text(
//                       "Disease Name",
//                       style: TextStyle(
//                         fontSize: 20,
//                         fontWeight: FontWeight.bold,
//                         color: Colors.black87,
//                       ),
//                     ),
//                     const SizedBox(height: 8),
//                     const Text(
//                       "Scientific Name",
//                       style: TextStyle(
//                         fontSize: 16,
//                         fontStyle: FontStyle.italic,
//                         color: Colors.black54,
//                       ),
//                     ),
//                     const SizedBox(height: 16),
//                     const Text(
//                       "Description",
//                       style: TextStyle(
//                         fontSize: 16,
//                         fontWeight: FontWeight.bold,
//                         color: Colors.black,
//                       ),
//                     ),
//                     const SizedBox(height: 4),
//                     const Text(
//                       "Your detected disease description will appear here after detection.",
//                       style: TextStyle(
//                         fontSize: 14,
//                         color: Colors.black54,
//                       ),
//                     ),
//                   ],
//                   const Spacer(),
//                   // Buttons
//                   Column(
//                     children: [
//                       SizedBox(
//                         width: double.infinity,
//                         child: ElevatedButton(
//                           onPressed: () {
//                             setState(() {
//                               _isDetected = true;
//                             });
//                             print("Detect button pressed!");
//                           },
//                           style: ElevatedButton.styleFrom(
//                             backgroundColor: const Color(0xFF277B53),
//                             padding: const EdgeInsets.symmetric(vertical: 16),
//                             shape: RoundedRectangleBorder(
//                               borderRadius: BorderRadius.circular(12),
//                             ),
//                           ),
//                           child: const Text(
//                             "Detect Disease",
//                             style: TextStyle(
//                               fontSize: 16,
//                               color: Colors.white,
//                             ),
//                           ),
//                         ),
//                       ),
//                       const SizedBox(height: 10),
//                     ],
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














