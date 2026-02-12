import { Component, OnInit, OnDestroy, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth';
import { AsistenciaService } from '../../services/asistencia';
import { FaceRecognitionService, FaceDescriptor } from '../../services/face-recognition.service';
import { NotificationService } from '../../services/notification.service';

type EstadoMarcaje = 'cargando' | 'no-enrollado' | 'ya-marco' | 'listo' | 'camara-activa' | 'verificando' | 'resultado';

@Component({
  selector: 'app-marcar-asistencia',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './marcar-asistencia.html',
  styleUrl: './marcar-asistencia.css',
})
export class MarcarAsistencia implements OnInit, OnDestroy {
  private auth = inject(AuthService);
  private asistenciaService = inject(AsistenciaService);
  private faceRecognition = inject(FaceRecognitionService);
  private notification = inject(NotificationService);
  private router = inject(Router);

  currentEmpleado: any;
  estado: EstadoMarcaje = 'cargando';
  videoStream: MediaStream | null = null;
  storedDescriptors: FaceDescriptor[] = [];

  isVerifying = false;
  verificationResult: { match: boolean; confidence: number } | null = null;
  marcajeRegistrado = false;
  livenessMessage = '';
  livenessConfirmed = false;

  asistenciaHoy: any = null;
  tipoMarcaje: 'entrada' | 'salida' = 'entrada';

  errorMessage = '';

  ngOnInit() {
    this.currentEmpleado = this.auth.getCurrentEmpleado();
    if (!this.currentEmpleado) {
      this.router.navigate(['/login']);
      return;
    }
    this.inicializar();
  }

  ngOnDestroy() {
    this.detenerCamara();
  }

  private async inicializar() {
    this.estado = 'cargando';
    try {
      // Verificar asistencia de hoy
      await this.verificarAsistenciaHoy();

      // Verificar enrollment
      const enrollCheck = await this.faceRecognition.checkEnrolled(this.currentEmpleado.id).toPromise();
      if (!enrollCheck?.enrolled) {
        this.estado = 'no-enrollado';
        return;
      }

      // Cargar descriptores
      const descriptors = await this.faceRecognition.getDescriptors(this.currentEmpleado.id).toPromise();
      this.storedDescriptors = descriptors || [];

      if (this.storedDescriptors.length === 0) {
        this.estado = 'no-enrollado';
        return;
      }

      // Cargar modelos
      await this.faceRecognition.loadModels();

      this.estado = 'listo';
    } catch (err: any) {
      this.errorMessage = err.error?.error || 'Error al inicializar el sistema de reconocimiento facial';
      this.estado = 'listo';
    }
  }

  private async verificarAsistenciaHoy() {
    try {
      const asistencias = await this.asistenciaService.getAsistenciaHoy().toPromise();
      const miAsistencia = asistencias?.find(
        (a: any) => a.empleadoId === this.currentEmpleado.id
      );

      if (miAsistencia) {
        this.asistenciaHoy = miAsistencia;
        if (miAsistencia.horaEntrada && miAsistencia.horaSalida) {
          this.estado = 'ya-marco';
          this.tipoMarcaje = 'entrada';
        } else if (miAsistencia.horaEntrada) {
          this.tipoMarcaje = 'salida';
        } else {
          this.tipoMarcaje = 'entrada';
        }
      }
    } catch {
      // Si falla, asumir que no ha marcado
    }
  }

  async abrirCamara() {
    this.errorMessage = '';
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 }
      });
      this.videoStream = stream;
      this.estado = 'camara-activa';

      this.livenessConfirmed = false;

      setTimeout(() => {
        const video = document.getElementById('marcajeVideo') as HTMLVideoElement;
        if (video) {
          video.srcObject = stream;
          video.onloadeddata = () => {
            const canvas = document.getElementById('marcajeOverlay') as HTMLCanvasElement;
            if (canvas) {
              this.faceRecognition.startOverlay(video, canvas, {
                detectLiveness: true,
                onLivenessChange: (confirmed) => {
                  this.livenessConfirmed = confirmed;
                }
              });
            }
          };
        }
      }, 100);
    } catch {
      this.errorMessage = 'No se pudo acceder a la camara. Verifique los permisos del navegador.';
      this.notification.error(this.errorMessage, 'Error de Camara');
    }
  }

  async verificarYMarcar() {
    if (this.isVerifying) return;

    const video = document.getElementById('marcajeVideo') as HTMLVideoElement;
    if (!video) {
      this.notification.error('Video no disponible', 'Error');
      return;
    }

    this.isVerifying = true;
    this.estado = 'verificando';
    this.livenessMessage = 'Verificando identidad...';

    try {
      const descriptor = await this.faceRecognition.detectSingleFace(video);

      if (!descriptor) {
        this.notification.warning(
          'No se detecto un rostro. Asegurese de tener buena iluminacion y mire a la camara.',
          'Rostro no detectado'
        );
        this.livenessMessage = '';
        this.isVerifying = false;
        this.estado = 'camara-activa';
        return;
      }

      const result = this.faceRecognition.verifyFace(descriptor, this.storedDescriptors);
      this.verificationResult = { match: result.match, confidence: result.confidence };

      if (result.match) {
        await this.registrarAsistencia();
        this.detenerCamara();
        this.estado = 'resultado';
      } else {
        this.notification.error(
          `Verificacion fallida. Confianza: ${result.confidence}%. El rostro no coincide con el registro.`,
          'No verificado'
        );
        this.isVerifying = false;
        this.estado = 'camara-activa';
      }
    } catch {
      this.notification.error('Error al procesar la verificacion facial', 'Error');
      this.isVerifying = false;
      this.estado = 'camara-activa';
    } finally {
      this.livenessMessage = '';
    }
  }

  private async registrarAsistencia() {
    try {
      const request = {
        empleadoId: this.currentEmpleado.id,
        tipo: this.tipoMarcaje,
        metodoVerificacion: 'facial'
      };

      const response = await this.asistenciaService.registrarAsistencia(request).toPromise();
      this.marcajeRegistrado = true;
      this.asistenciaHoy = response;
      this.notification.success(
        `${this.tipoMarcaje === 'entrada' ? 'Entrada' : 'Salida'} registrada exitosamente con verificacion facial`,
        'Asistencia Registrada'
      );
    } catch (err: any) {
      const msg = err.error?.error || 'Error al registrar asistencia';
      this.notification.error(msg, 'Error');
      this.verificationResult = null;
    } finally {
      this.isVerifying = false;
    }
  }

  detenerCamara() {
    this.faceRecognition.stopOverlay();
    if (this.videoStream) {
      this.videoStream.getTracks().forEach(track => track.stop());
      this.videoStream = null;
    }
  }

  volverAlDashboard() {
    this.detenerCamara();
    this.router.navigate(['/dashboard']);
  }

  reintentar() {
    this.verificationResult = null;
    this.marcajeRegistrado = false;
    this.errorMessage = '';
    this.inicializar();
  }
}
