import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ApiConfigService } from './api-config.service';
import * as faceapi from 'face-api.js';

export interface FaceDescriptor {
  id?: number;
  empleadoId: number;
  descriptor: number[];
  comentario?: string;
}

export interface FaceEnrollRequest {
  empleadoId: number;
  descriptors: number[][];
  comentario?: string;
}

export interface VerificationResult {
  match: boolean;
  distance: number;
  confidence: number;
  bestMatch?: FaceDescriptor;
}

@Injectable({
  providedIn: 'root'
})
export class FaceRecognitionService {
  private http = inject(HttpClient);
  private apiConfig = inject(ApiConfigService);
  private modelsLoaded = false;
  private modelsLoading = false;

  private readonly VERIFICATION_THRESHOLD = 0.6;

  private get apiUrl() {
    return `${this.apiConfig.apiUrl}/face`;
  }

  async loadModels(): Promise<void> {
    if (this.modelsLoaded) return;
    if (this.modelsLoading) {
      // Esperar a que termine la carga en curso
      while (this.modelsLoading) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      return;
    }

    this.modelsLoading = true;
    try {
      const MODEL_URL = '/models';
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL)
      ]);
      this.modelsLoaded = true;
    } finally {
      this.modelsLoading = false;
    }
  }

  async detectSingleFace(input: HTMLVideoElement | HTMLCanvasElement): Promise<Float32Array | null> {
    if (!this.modelsLoaded) {
      await this.loadModels();
    }

    const detection = await faceapi
      .detectSingleFace(input, new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.5 }))
      .withFaceLandmarks()
      .withFaceDescriptor();

    return detection ? detection.descriptor : null;
  }

  verifyFace(
    capturedDescriptor: Float32Array,
    storedDescriptors: FaceDescriptor[],
    threshold: number = this.VERIFICATION_THRESHOLD
  ): VerificationResult {
    if (!storedDescriptors || storedDescriptors.length === 0) {
      return { match: false, distance: Infinity, confidence: 0 };
    }

    let bestDistance = Infinity;
    let bestMatch: FaceDescriptor | undefined;

    for (const stored of storedDescriptors) {
      const distance = faceapi.euclideanDistance(
        Array.from(capturedDescriptor),
        stored.descriptor
      );
      if (distance < bestDistance) {
        bestDistance = distance;
        bestMatch = stored;
      }
    }

    const confidence = Math.max(0, Math.round((1 - bestDistance) * 100));

    return {
      match: bestDistance < threshold,
      distance: bestDistance,
      confidence,
      bestMatch
    };
  }

  // --- Liveness Detection (basic blink detection via eye landmarks) ---

  async detectLiveness(video: HTMLVideoElement, timeoutMs: number = 5000): Promise<boolean> {
    if (!this.modelsLoaded) {
      await this.loadModels();
    }

    const startTime = Date.now();
    let blinkDetected = false;
    let prevEAR = -1;
    const EAR_THRESHOLD = 0.22;
    const EAR_CONSEC_FRAMES = 2;
    let belowThresholdCount = 0;

    while (Date.now() - startTime < timeoutMs && !blinkDetected) {
      const detection = await faceapi
        .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 }))
        .withFaceLandmarks();

      if (detection) {
        const landmarks = detection.landmarks;
        const leftEye = landmarks.getLeftEye();
        const rightEye = landmarks.getRightEye();

        const earLeft = this.computeEAR(leftEye);
        const earRight = this.computeEAR(rightEye);
        const ear = (earLeft + earRight) / 2;

        if (ear < EAR_THRESHOLD) {
          belowThresholdCount++;
        } else {
          if (belowThresholdCount >= EAR_CONSEC_FRAMES && prevEAR >= EAR_THRESHOLD) {
            blinkDetected = true;
          }
          belowThresholdCount = 0;
        }
        prevEAR = ear;
      }

      await new Promise(resolve => setTimeout(resolve, 150));
    }

    return blinkDetected;
  }

  private computeEAR(eye: faceapi.Point[]): number {
    // Eye Aspect Ratio (EAR) formula
    // EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    // Using 6-point eye landmarks: indices 0-5
    const dist = (a: faceapi.Point, b: faceapi.Point) =>
      Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));

    const vertical1 = dist(eye[1], eye[5]);
    const vertical2 = dist(eye[2], eye[4]);
    const horizontal = dist(eye[0], eye[3]);

    return (vertical1 + vertical2) / (2.0 * horizontal);
  }

  // --- Backend API calls ---

  enrollFace(request: FaceEnrollRequest): Observable<any> {
    return this.http.post(`${this.apiUrl}/enroll`, request);
  }

  getDescriptors(empleadoId: number): Observable<FaceDescriptor[]> {
    return this.http.get<FaceDescriptor[]>(`${this.apiUrl}/descriptors/${empleadoId}`);
  }

  checkEnrolled(empleadoId: number): Observable<{ enrolled: boolean }> {
    return this.http.get<{ enrolled: boolean }>(`${this.apiUrl}/enrolled/${empleadoId}`);
  }

  deleteDescriptors(empleadoId: number): Observable<any> {
    return this.http.delete(`${this.apiUrl}/descriptors/${empleadoId}`);
  }
}
