import { useEffect, useRef, useState } from "react";
import "./Brain3DViewer.css";

export default function Brain3DViewer({ findings }) {
  const canvasRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const rendererRef = useRef(null);
  const brainMeshRef = useRef(null);
  const animateRef = useRef(null);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const loadAttemptedRef = useRef(false);

  useEffect(() => {
    let timeoutId;
    
    // Prevent loading twice in React StrictMode
    if (loadAttemptedRef.current) {
      console.log("Load already attempted, skipping");
      return;
    }
    loadAttemptedRef.current = true;
    
    // Load Three.js only
    const loadLibraries = () => {
      // Check if Three.js is already loaded
      if (window.THREE) {
        console.log("✓ Three.js already loaded");
        timeoutId = setTimeout(() => {
          initScene();
        }, 100);
        return;
      }

      const script1 = document.createElement("script");
      script1.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
      script1.async = true;
      script1.crossOrigin = "anonymous";
      
      script1.onload = () => {
        console.log("✓ Three.js loaded");
        
        // Load GLTFLoader from unpkg as alternative
        const script2 = document.createElement("script");
        script2.src = "https://unpkg.com/three@r128/examples/js/loaders/GLTFLoader.js";
        script2.async = false;
        script2.crossOrigin = "anonymous";
        
        script2.onload = () => {
          console.log("✓ GLTFLoader loaded");
          timeoutId = setTimeout(() => {
            initScene();
          }, 200);
        };
        
        script2.onerror = () => {
          console.warn("⚠ GLTFLoader CDN failed, will use fallback");
          timeoutId = setTimeout(() => {
            initScene();
          }, 200);
        };
        
        document.head.appendChild(script2);
      };
      
      script1.onerror = () => {
        console.error("✗ Failed to load Three.js");
        setIsLoading(false);
      };
      
      document.head.appendChild(script1);
    };

    loadLibraries();

    return () => {
      if (timeoutId) clearTimeout(timeoutId);
      if (animateRef.current) {
        cancelAnimationFrame(animateRef.current);
      }
      if (rendererRef.current && canvasRef.current) {
        try {
          const element = rendererRef.current.domElement;
          if (element && element.parentNode === canvasRef.current) {
            canvasRef.current.removeChild(element);
          }
        } catch (e) {
          console.error("Cleanup error:", e);
        }
      }
    };
  }, []);

  const initScene = () => {
    if (!window.THREE) {
      console.error("Three.js not available");
      setIsLoading(false);
      return;
    }

    const THREE = window.THREE;
    const canvas = canvasRef.current;
    
    if (!canvas || !canvas.clientWidth) {
      console.error("Canvas not ready");
      setIsLoading(false);
      return;
    }

    console.log("Initializing scene...");
    const hasGLTFLoader = typeof window.THREE !== 'undefined' && 
                          typeof window.THREE.GLTFLoader !== 'undefined' &&
                          window.THREE.GLTFLoader !== undefined;
    console.log("GLTFLoader available:", hasGLTFLoader);

    try {
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf0f4ff);

      const camera = new THREE.PerspectiveCamera(
        75,
        canvas.clientWidth / canvas.clientHeight,
        0.1,
        1000
      );
      camera.position.z = 2.5;

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(canvas.clientWidth, canvas.clientHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      canvas.appendChild(renderer.domElement);
      rendererRef.current = renderer;

      // Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
      scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.9);
      directionalLight.position.set(5, 5, 5);
      scene.add(directionalLight);

      const pointLight = new THREE.PointLight(0xff6b6b, 0.4);
      pointLight.position.set(-5, 5, 5);
      scene.add(pointLight);

      // Try to load GLB model if GLTFLoader is available
      if (hasGLTFLoader) {
        console.log("Loading brain model with GLTFLoader...");
        const loader = new window.THREE.GLTFLoader();
        
        loader.load(
          "/brain-model.glb",
          (gltf) => {
            console.log("✓ Brain model loaded successfully");
            const model = gltf.scene;
            
            model.scale.set(1, 1, 1);
            model.position.set(0, 0, 0);
            
            model.traverse((node) => {
              if (node.isMesh) {
                node.material = new THREE.MeshStandardMaterial({
                  color: 0xffb3ba,
                  metalness: 0.2,
                  roughness: 0.8,
                  emissive: 0xffb3ba,
                  emissiveIntensity: 0.1,
                });
              }
            });
            
            scene.add(model);
            brainMeshRef.current = model;
            addProblemAreaHighlights(scene, findings);
            startAnimation(scene, camera, renderer);
            setIsLoading(false);
          },
          undefined,
          (error) => {
            console.error("✗ Model load failed:", error.message);
            console.log("Falling back to procedural brain");
            createFallbackBrain(scene);
            addProblemAreaHighlights(scene, findings);
            startAnimation(scene, camera, renderer);
            setIsLoading(false);
          }
        );
      } else {
        console.log("GLTFLoader not available, using procedural fallback");
        createFallbackBrain(scene);
        addProblemAreaHighlights(scene, findings);
        startAnimation(scene, camera, renderer);
        setIsLoading(false);
      }

      const handleResize = () => {
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
      };

      window.addEventListener("resize", handleResize);
    } catch (error) {
      console.error("Scene init error:", error);
      setIsLoading(false);
    }
  };

  const startAnimation = (scene, camera, renderer) => {
    const animate = () => {
      animateRef.current = requestAnimationFrame(animate);

      if (!isDragging && brainMeshRef.current) {
        brainMeshRef.current.rotation.x += 0.001;
        brainMeshRef.current.rotation.y += 0.002;
      }

      if (brainMeshRef.current) {
        brainMeshRef.current.rotation.x +=
          (rotation.x - brainMeshRef.current.rotation.x) * 0.05;
        brainMeshRef.current.rotation.y +=
          (rotation.y - brainMeshRef.current.rotation.y) * 0.05;
      }

      renderer.render(scene, camera);
    };

    animate();
  };

  const createFallbackBrain = (scene) => {
    if (!window.THREE) return;
    
    const THREE = window.THREE;
    console.log("Using fallback brain");
    
    const geometry = new THREE.IcosahedronGeometry(1, 6);
    const positionAttribute = geometry.getAttribute("position");
    const positions = positionAttribute.array;
    
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i];
      const y = positions[i + 1];
      const z = positions[i + 2];
      
      const noise1 = Math.sin(x * 5) * Math.cos(y * 5) * Math.sin(z * 5);
      const noise2 = Math.sin(x * 3) * Math.sin(y * 3) * Math.cos(z * 3);
      const totalNoise = noise1 * 0.05 + noise2 * 0.04;
      
      const length = Math.sqrt(x * x + y * y + z * z);
      const scale = 1 + totalNoise / length;
      
      positions[i] = x * scale;
      positions[i + 1] = y * scale;
      positions[i + 2] = z * scale;
    }
    
    positionAttribute.needsUpdate = true;
    geometry.computeVertexNormals();
    
    const material = new THREE.MeshStandardMaterial({
      color: 0xffb3ba,
      metalness: 0.1,
      roughness: 0.8,
    });
    
    const brain = new THREE.Mesh(geometry, material);
    scene.add(brain);
    brainMeshRef.current = brain;
  };

  const addProblemAreaHighlights = (scene, findings) => {
    if (!findings || !window.THREE) return;

    const THREE = window.THREE;

    findings.forEach((finding, index) => {
      if (
        finding.status &&
        (finding.status.toLowerCase().includes("abnormal") ||
          finding.status.toLowerCase().includes("mild"))
      ) {
        const isAbnormal = finding.status.toLowerCase().includes("abnormal");
        const color = isAbnormal ? 0xff6b6b : 0xffc107;
        
        const highlightGeometry = new THREE.SphereGeometry(0.25, 32, 32);
        const highlightMaterial = new THREE.MeshStandardMaterial({
          color: color,
          emissive: color,
          emissiveIntensity: 0.8,
          metalness: 0.5,
          roughness: 0.3,
        });

        const highlight = new THREE.Mesh(highlightGeometry, highlightMaterial);

        const angleXZ = (index / findings.length) * Math.PI * 2;
        const angleY = ((index % 2) * Math.PI) / 2;
        const radius = 1.4;
        
        highlight.position.x = Math.cos(angleXZ) * radius * Math.cos(angleY);
        highlight.position.y = Math.sin(angleY) * radius;
        highlight.position.z = Math.sin(angleXZ) * radius * Math.cos(angleY);

        scene.add(highlight);
      }
    });
  };

  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;

    const deltaX = e.clientX - dragStart.x;
    const deltaY = e.clientY - dragStart.y;

    setRotation((prev) => ({
      x: prev.x + deltaY * 0.005,
      y: prev.y + deltaX * 0.005,
    }));

    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  return (
    <div className="brain-3d-viewer">
      {isLoading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>Loading 3D Brain Model...</p>
        </div>
      )}
      <div
        ref={canvasRef}
        className="brain-canvas"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
      <div className="brain-controls">
        <p className="control-hint">🖱️ Click and drag to rotate</p>
        <div className="legend">
          <div className="legend-item">
            <div className="legend-color abnormal"></div>
            <span>Abnormal</span>
          </div>
          <div className="legend-item">
            <div className="legend-color mild"></div>
            <span>Mild Changes</span>
          </div>
        </div>
      </div>
    </div>
  );
}
