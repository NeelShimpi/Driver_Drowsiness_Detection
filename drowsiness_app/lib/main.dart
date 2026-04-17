import 'dart:async';
import 'dart:convert';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final firstCamera = cameras.firstWhere(
    (camera) => camera.lensDirection == CameraLensDirection.front,
    orElse: () => cameras.first,
  );

  runApp(MyApp(camera: firstCamera));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  const MyApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Drowsiness Detection',
      theme: ThemeData.dark().copyWith(
        primaryColor: const Color(0xFF00E5FF),
        scaffoldBackgroundColor: Colors.black,
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00E5FF),
          secondary: Color(0xFFFF3366),
        ),
      ),
      home: DrowsinessScreen(camera: camera),
      debugShowCheckedModeBanner: false,
    );
  }
}

class DrowsinessScreen extends StatefulWidget {
  final CameraDescription camera;

  const DrowsinessScreen({super.key, required this.camera});

  @override
  State<DrowsinessScreen> createState() => _DrowsinessScreenState();
}

class _DrowsinessScreenState extends State<DrowsinessScreen>
    with SingleTickerProviderStateMixin {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;

  Timer? _processingTimer;
  bool _isProcessing = false;

  String _backendUrl = 'http://127.0.0.1:5000/api/detect';
  final TextEditingController _urlController = TextEditingController();

  Map<String, dynamic>? _lastResult;
  bool _isDrowsy = false;
  int _facesDetected = 0;

  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _urlController.text = _backendUrl;
    
    // Setup pulse animation for alerts
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );
    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.2).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    _initCamera();
  }

  void _initCamera() {
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    _initializeControllerFuture = _controller.initialize().then((_) {
      _startDetectionLoop();
    });
  }

  void _startDetectionLoop() {
    _processingTimer = Timer.periodic(const Duration(milliseconds: 500), (timer) async {
      if (_isProcessing) return;
      _isProcessing = true;

      try {
        final image = await _controller.takePicture();
        await _processImage(image);
      } catch (e) {
        debugPrint('Error capturing frame: $e');
      } finally {
        _isProcessing = false;
        if (mounted) setState(() {});
      }
    });
  }

  Future<void> _processImage(XFile file) async {
    try {
      final bytes = await file.readAsBytes();
      final base64Image = base64Encode(bytes);

      final response = await http.post(
        Uri.parse(_backendUrl),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'image': base64Image}),
      ).timeout(const Duration(seconds: 2));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (mounted) {
          setState(() {
            _lastResult = data;
            _facesDetected = data['faces_detected'] ?? 0;
            _isDrowsy = data['overall_drowsy'] ?? false;
            
            if (_isDrowsy) {
              _pulseController.repeat(reverse: true);
            } else {
              _pulseController.stop();
              _pulseController.value = 1.0;
            }
          });
        }
      }
    } catch (e) {
      debugPrint('Backend connection error: $e');
    }
  }

  @override
  void dispose() {
    _processingTimer?.cancel();
    _controller.dispose();
    _urlController.dispose();
    _pulseController.dispose();
    super.dispose();
  }

  // Helper method to create a frosted glass effect
  Widget _buildFrostedGlass({
    required Widget child,
    double borderRadius = 16,
    EdgeInsets padding = const EdgeInsets.all(16),
  }) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(borderRadius),
      child: BackdropFilter(
        filter: ui.ImageFilter.blur(sigmaX: 10.0, sigmaY: 10.0),
        child: Container(
          padding: padding,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.08),
            borderRadius: BorderRadius.circular(borderRadius),
            border: Border.all(
              color: Colors.white.withOpacity(0.2),
              width: 1.5,
            ),
          ),
          child: child,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    // Determine dynamic colors for UI alerts
    Color statusColor = const Color(0xFF00E5FF); // Cyber Cyan
    String statusText = 'System Monitoring';
    
    if (_facesDetected == 0) {
      statusColor = Colors.orangeAccent;
      statusText = 'No Face Detected';
    } else if (_isDrowsy) {
      statusColor = const Color(0xFFFF3366); // Neon Red
      statusText = 'DROWSINESS ALERT';
    } else {
      statusColor = const Color(0xFF00E5FF);
      statusText = 'Driver Alert & Focused';
    }

    return Scaffold(
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            return Stack(
              fit: StackFit.expand,
              children: [
                // 1. Full Screen Camera Background
                FittedBox(
                  fit: BoxFit.cover,
                  child: SizedBox(
                    width: _controller.value.previewSize?.height ?? 1,
                    height: _controller.value.previewSize?.width ?? 1,
                    child: CameraPreview(_controller),
                  ),
                ),
                
                // 2. Dimming Vignette Overlay
                Container(
                  decoration: BoxDecoration(
                    gradient: RadialGradient(
                      colors: [
                        Colors.transparent,
                        _isDrowsy ? Colors.red.withOpacity(0.4) : Colors.black.withOpacity(0.6),
                      ],
                      center: Alignment.center,
                      radius: 0.8,
                    ),
                  ),
                ),

                // 3. Status Pill (Top)
                Positioned(
                  top: 60,
                  left: 20,
                  right: 20,
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 300),
                    curve: Curves.easeOutCubic,
                    child: _buildFrostedGlass(
                      borderRadius: 30,
                      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 20),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            _isDrowsy ? Icons.warning_amber_rounded : Icons.shield_outlined,
                            color: statusColor,
                            size: 28,
                          ),
                          const SizedBox(width: 12),
                          Text(
                            statusText.toUpperCase(),
                            style: TextStyle(
                              color: statusColor,
                              fontSize: 18,
                              fontWeight: FontWeight.w800,
                              letterSpacing: 1.2,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),

                // 4. Central HUD Elements
                Center(
                  child: ScaleTransition(
                    scale: _isDrowsy ? _pulseAnimation : const AlwaysStoppedAnimation(1.0),
                    child: Container(
                      width: 280,
                      height: 280,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: statusColor.withOpacity(0.5),
                          width: 2,
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: statusColor.withOpacity(_isDrowsy ? 0.3 : 0.0),
                            blurRadius: 30,
                            spreadRadius: 10,
                          ),
                        ],
                      ),
                      child: Center(
                        child: Container(
                          width: 20,
                          height: 20,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: statusColor.withOpacity(0.8),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),

                // 5. Stats Bar (Bottom)
                Positioned(
                  bottom: 40,
                  left: 20,
                  right: 20,
                  child: _buildFrostedGlass(
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _buildStatItem('Faces', '$_facesDetected', Icons.face),
                        if (_lastResult != null && _lastResult!['processing_time_ms'] != null)
                          _buildStatItem(
                            'Latency', 
                            '${_lastResult!['processing_time_ms'].toStringAsFixed(0)}ms', 
                            Icons.speed
                          ),
                      ],
                    ),
                  ),
                ),

                // 6. Settings Floating Button
                Positioned(
                  top: 130,
                  right: 20,
                  child: Material(
                    color: Colors.transparent,
                    child: InkWell(
                      borderRadius: BorderRadius.circular(20),
                      onTap: _showSettingsDialog,
                      child: _buildFrostedGlass(
                        borderRadius: 20,
                        padding: const EdgeInsets.all(12),
                        child: const Icon(Icons.tune, color: Colors.white70),
                      ),
                    ),
                  ),
                ),
              ],
            );
          } else {
            return const Center(child: CircularProgressIndicator(color: Color(0xFF00E5FF)));
          }
        },
      ),
    );
  }

  Widget _buildStatItem(String label, String value, IconData icon) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, color: Colors.white54, size: 24),
        const SizedBox(height: 8),
        Text(
          value,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 22,
            fontWeight: FontWeight.bold,
          ),
        ),
        Text(
          label.toUpperCase(),
          style: const TextStyle(
            color: Colors.white54,
            fontSize: 10,
            letterSpacing: 1.5,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }

  void _showSettingsDialog() {
    showDialog(
      context: context,
      builder: (context) => Dialog(
        backgroundColor: Colors.transparent,
        child: _buildFrostedGlass(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Network Settings',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 16),
              const Text(
                'Backend API URL:',
                style: TextStyle(color: Colors.white70),
              ),
              const SizedBox(height: 8),
              TextField(
                controller: _urlController,
                style: const TextStyle(color: Colors.white),
                decoration: InputDecoration(
                  filled: true,
                  fillColor: Colors.black45,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                    borderSide: BorderSide.none,
                  ),
                  hintText: 'http://127.0.0.1:5000/api/detect',
                  hintStyle: const TextStyle(color: Colors.white24),
                ),
              ),
              const SizedBox(height: 24),
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('Cancel', style: TextStyle(color: Colors.white54)),
                  ),
                  const SizedBox(width: 8),
                  ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF00E5FF),
                      foregroundColor: Colors.black,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                    onPressed: () {
                      setState(() {
                        _backendUrl = _urlController.text.trim();
                      });
                      Navigator.pop(context);
                    },
                    child: const Text('Save Connection', style: TextStyle(fontWeight: FontWeight.bold)),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
