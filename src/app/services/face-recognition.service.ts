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

  // --- Face Detection Overlay con Liveness pasivo (varianza EAR) ---

  private overlayInterval: any = null;
  livenessConfirmed = false;
  private _earHistory: number[] = [];
  private readonly EAR_HISTORY_SIZE = 12;
  private readonly EAR_VARIANCE_THRESHOLD = 0.0004;
  private _onLivenessChange: ((confirmed: boolean) => void) | null = null;

  startOverlay(
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    options?: { detectLiveness?: boolean; onLivenessChange?: (confirmed: boolean) => void }
  ): void {
    this.stopOverlay();
    this.livenessConfirmed = false;
    this._earHistory = [];
    this._onLivenessChange = options?.onLivenessChange || null;
    const detectLiveness = options?.detectLiveness ?? false;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    this.overlayInterval = setInterval(async () => {
      if (video.paused || video.ended) return;

      const displayWidth = video.clientWidth;
      const displayHeight = video.clientHeight;
      canvas.width = displayWidth;
      canvas.height = displayHeight;

      const detection = await faceapi
        .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 }))
        .withFaceLandmarks();

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (detection) {
        const scaleX = displayWidth / video.videoWidth;
        const scaleY = displayHeight / video.videoHeight;
        const box = detection.detection.box;
        const mirroredX = displayWidth - (box.x * scaleX) - (box.width * scaleX);

        const color = (!detectLiveness || this.livenessConfirmed) ? '#00ff88' : '#00d4ff';

        // Rectángulo
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.shadowColor = color;
        ctx.shadowBlur = 6;
        ctx.strokeRect(mirroredX, box.y * scaleY, box.width * scaleX, box.height * scaleY);
        ctx.shadowBlur = 0;

        // Esquinas decorativas
        const cornerLen = 15;
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        const bx = mirroredX, by = box.y * scaleY;
        const bw = box.width * scaleX, bh = box.height * scaleY;

        ctx.beginPath(); ctx.moveTo(bx, by + cornerLen); ctx.lineTo(bx, by); ctx.lineTo(bx + cornerLen, by); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(bx + bw - cornerLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + cornerLen); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(bx, by + bh - cornerLen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + cornerLen, by + bh); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(bx + bw - cornerLen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - cornerLen); ctx.stroke();

        // Landmarks
        const landmarks = detection.landmarks.positions;
        ctx.fillStyle = color;
        for (const point of landmarks) {
          const mx = displayWidth - (point.x * scaleX);
          ctx.beginPath();
          ctx.arc(mx, point.y * scaleY, 1.5, 0, 2 * Math.PI);
          ctx.fill();
        }

        // Etiqueta
        const score = Math.round(detection.detection.score * 100);
        ctx.font = '12px Arial';
        ctx.fillStyle = color;
        ctx.fillText(`Rostro ${score}%`, mirroredX, by - 6);

        // --- Liveness por varianza de EAR ---
        if (detectLiveness && !this.livenessConfirmed) {
          const leftEye = detection.landmarks.getLeftEye();
          const rightEye = detection.landmarks.getRightEye();
          const earLeft = this.computeEAR(leftEye);
          const earRight = this.computeEAR(rightEye);
          const ear = (earLeft + earRight) / 2;

          this._earHistory.push(ear);
          if (this._earHistory.length > this.EAR_HISTORY_SIZE) {
            this._earHistory.shift();
          }

          // Cuando tenemos suficientes muestras, calcular varianza
          if (this._earHistory.length >= this.EAR_HISTORY_SIZE) {
            const mean = this._earHistory.reduce((a, b) => a + b, 0) / this._earHistory.length;
            const variance = this._earHistory.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / this._earHistory.length;

            if (variance > this.EAR_VARIANCE_THRESHOLD) {
              this.livenessConfirmed = true;
              if (this._onLivenessChange) {
                this._onLivenessChange(true);
              }
            }
          }
        }

        // Badge de liveness en el canvas
        if (detectLiveness) {
          const badgeY = by + bh + 20;
          if (this.livenessConfirmed) {
            ctx.font = 'bold 13px Arial';
            ctx.fillStyle = '#00ff88';
            ctx.fillText('Persona real', mirroredX, badgeY);
          } else {
            const progress = Math.min(this._earHistory.length, this.EAR_HISTORY_SIZE);
            ctx.font = '11px Arial';
            ctx.fillStyle = '#ffcc00';
            ctx.fillText(`Analizando... (${progress}/${this.EAR_HISTORY_SIZE})`, mirroredX, badgeY);
          }
        }
      }
    }, 150);
  }

  stopOverlay(): void {
    if (this.overlayInterval) {
      clearInterval(this.overlayInterval);
      this.overlayInterval = null;
    }
    this._onLivenessChange = null;
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
