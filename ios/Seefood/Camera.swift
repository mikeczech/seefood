//
//  CameraView.swift
//  Seefood
//
//  Created by Mike Czech on 29.12.19.
//  Copyright © 2019 Mike Czech. All rights reserved.
//

import SwiftUI

import AVFoundation
import UIKit

final class CameraViewController: UIViewController {
    
    enum SessionSetupResult {
        case success
        case notAuthorized
        case configurationFailed
    }
    
    var delegate: AVCapturePhotoCaptureDelegate?
    
    let session = AVCaptureSession()
    let photoOutput = AVCapturePhotoOutput()
    let sessionQueue = DispatchQueue(label: "session queue",
                                     attributes: [],
                                     target: nil)
    
    var previewLayer: AVCaptureVideoPreviewLayer!
    var videoDeviceInput: AVCaptureDeviceInput!
    var setupResult: SessionSetupResult = .success
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        checkAuthorization()
        
        sessionQueue.async { [unowned self] in
            self.configureSession()
        }
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        sessionQueue.async {
            switch self.setupResult {
            case .success:
                DispatchQueue.main.async { [unowned self] in
                    self.previewLayer = AVCaptureVideoPreviewLayer(session: self.session)
                    self.previewLayer.frame = self.view.layer.bounds
                    self.previewLayer.videoGravity = .resizeAspectFill
                    self.view.layer.addSublayer(self.previewLayer)
                    
                    let captureButton = UIButton(frame: CGRect(x: self.view.center.x - 40, y: self.view.center.y - 40 + 260, width: 80, height: 80))
                    captureButton.layer.cornerRadius = 0.5 * captureButton.bounds.size.width
                    captureButton.clipsToBounds = true
                    captureButton.layer.borderColor = UIColor.gray.cgColor
                    captureButton.layer.borderWidth = 6
                    captureButton.backgroundColor = UIColor.white
                    captureButton.alpha = 0.6
                    captureButton.addTarget(self, action:#selector(self.capturePhoto), for: .touchUpInside)
                    
                    self.view.addSubview(captureButton)
                    self.session.startRunning()
                }
            case .notAuthorized:
                DispatchQueue.main.async { [unowned self] in
                    let changePrivacySetting = "AVCam doesn't have permission to use the camera, please change privacy settings"
                    let message = NSLocalizedString(changePrivacySetting, comment: "Alert message when the user has denied access to the camera")
                    let alertController = UIAlertController(title: "AVCam", message: message, preferredStyle: .alert)
                    
                    alertController.addAction(UIAlertAction(title: NSLocalizedString("OK", comment: "Alert OK button"),
                                                            style: .cancel,
                                                            handler: nil))
                    
                    alertController.addAction(UIAlertAction(title: NSLocalizedString("Settings", comment: "Alert button to open Settings"),
                                                            style: .default,
                                                            handler: { _ in
                                                                UIApplication.shared.open(URL(string: UIApplication.openSettingsURLString)!, options: [:], completionHandler: nil)
                                                            }
                    ))
                    
                    self.present(alertController, animated: true, completion: nil)
                }
            case .configurationFailed:
                DispatchQueue.main.async { [unowned self] in
                    let alertMsg = "Alert message when something goes wrong during capture session configuration"
                    let message = NSLocalizedString("Unable to capture media", comment: alertMsg)
                    let alertController = UIAlertController(title: "AVCam", message: message, preferredStyle: .alert)

                    alertController.addAction(UIAlertAction(title: NSLocalizedString("OK", comment: "Alert OK button"),
                                                            style: .cancel,
                                                            handler: nil))
                    
                    self.present(alertController, animated: true, completion: nil)
                }
            }
        }
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        sessionQueue.async { [unowned self] in
            if self.setupResult == .success {
                self.session.stopRunning()
            }
        }
    }
    
    private func checkAuthorization() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            break
        case .notDetermined:
            sessionQueue.suspend()
            AVCaptureDevice.requestAccess(for: .video) {[unowned self] granted in
                if !granted {
                    self.setupResult = .notAuthorized
                }
                self.sessionQueue.resume()
            }
        default:
            setupResult = .notAuthorized
        }
    }
    
    private func configureSession() {
        if setupResult != .success {
            return
        }
        
        session.beginConfiguration()
        session.sessionPreset = AVCaptureSession.Preset.photo
        
        // add input
        do {
            let videoDeviceInput = try AVCaptureDeviceInput(device: AVCaptureDevice.default(for: .video)!)
            if session.canAddInput(videoDeviceInput) {
                session.addInput(videoDeviceInput)
                self.videoDeviceInput = videoDeviceInput
            } else {
                print("Could not add video device input to the session")
                setupResult = .configurationFailed
                session.commitConfiguration()
                return
            }
            
        } catch {
            print("Could not create video device input \(error)")
            setupResult = .configurationFailed
            session.commitConfiguration()
            return
        }
        
        // add output
        if session.canAddOutput(photoOutput) {
            session.addOutput(photoOutput)
        } else {
            print("Could not add photo output to the session")
            setupResult = .configurationFailed
            session.commitConfiguration()
            return
        }
        
        session.commitConfiguration()
    }
    
    @objc func capturePhoto(sender: UIButton!) {
        let photoSettings = AVCapturePhotoSettings()
        if self.videoDeviceInput.device.isFlashAvailable {
            photoSettings.flashMode = .auto
        }
        
        if let firstAvailablePreviewPhotoPixelFormatTypes = photoSettings.availablePreviewPhotoPixelFormatTypes.first {
            photoSettings.previewPhotoFormat = [kCVPixelBufferPixelFormatTypeKey as String: firstAvailablePreviewPhotoPixelFormatTypes]
        }
        
        
        let deviceOrientation = UIDevice.current.orientation
        guard let photoOutputConnection = photoOutput.connection(with: AVMediaType.video) else {fatalError("Unable to establish input>output connection")}
        guard let videoOrientation = deviceOrientation.getAVCaptureVideoOrientationFromDevice()  else { return }
        photoOutputConnection.videoOrientation = videoOrientation
        
        photoOutput.capturePhoto(with: photoSettings, delegate: self.delegate!)
    }

}

extension UIDeviceOrientation {
    func getAVCaptureVideoOrientationFromDevice() -> AVCaptureVideoOrientation? {
        // return AVCaptureVideoOrientation from device
        switch self {
        case UIDeviceOrientation.portrait: return AVCaptureVideoOrientation.portrait
        case UIDeviceOrientation.portraitUpsideDown: return AVCaptureVideoOrientation.portraitUpsideDown
        case UIDeviceOrientation.landscapeLeft: return AVCaptureVideoOrientation.landscapeRight // TODO why reversed?
        case UIDeviceOrientation.landscapeRight: return AVCaptureVideoOrientation.landscapeLeft
        case UIDeviceOrientation.faceDown: return AVCaptureVideoOrientation.portrait // not so sure about this one
        case UIDeviceOrientation.faceUp: return AVCaptureVideoOrientation.portrait // not so sure about this one
        case UIDeviceOrientation.unknown: return nil
        @unknown default:
            fatalError()
        }
    }
}

struct CameraViewControllerRepresentable: UIViewControllerRepresentable {

    public typealias UIViewControllerType = CameraViewController
    
    @Binding var inputImage: Image?
    @Binding var createEntry: Bool
        
    let cameraViewController = CameraViewController()

    class Coordinator : NSObject, AVCapturePhotoCaptureDelegate {
        var parent: CameraViewControllerRepresentable

        init(_ parent: CameraViewControllerRepresentable) {
            self.parent = parent
        }
        
        func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
            guard let data = photo.fileDataRepresentation(),
                  let image = UIImage(data: data) else {
                    return
                  }

            parent.inputImage = Image(uiImage: image)
            parent.createEntry = true
        }
    }
    
    func makeUIViewController(context: UIViewControllerRepresentableContext<CameraViewControllerRepresentable>) -> CameraViewController {
        cameraViewController.delegate = context.coordinator
        return cameraViewController
    }
    
    func updateUIViewController(_ uiViewController: CameraViewController, context: UIViewControllerRepresentableContext<CameraViewControllerRepresentable>) {
    }
    
    func makeCoordinator() -> CameraViewControllerRepresentable.Coordinator {
        Coordinator(self)
    }

}
