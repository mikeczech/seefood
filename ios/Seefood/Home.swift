//
//  Home.swift
//  Seefood
//
//  Created by Mike Czech on 29.12.19.
//  Copyright © 2019 Mike Czech. All rights reserved.
//

import SwiftUI

struct Home: View {
    
    @State private var showCamera = false
    @State private var image: Image?
    @State private var inputImage: UIImage?
        
        
    var body: some View {
        let cameraController = CameraViewControllerRepresentable(image: self.$inputImage)
        
        return VStack {
            HStack {
                Spacer()
                Text("All Time")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding(.top, 50.0)
                    .padding(.bottom, 20.0)
                Spacer()
            }
            .background(Color.blue)
            
            if self.showCamera {
                cameraController
            } else {
                image?
                    .resizable()
                    .scaledToFit()
            }
            
            Spacer()
            HStack(alignment: .top) {
                VStack {
                    Button(action: {
                        self.showCamera = false
                    }) {
                        Image(systemName: "house.fill")
                            .imageScale(.large)
                            .foregroundColor(.white)
                    }
                    Text("Home")
                        .font(.caption)
                        .padding(5)
                        .foregroundColor(.white)
                }
                .padding()
                .frame(width: 80.0, height: 80.0)
                Spacer()
                VStack {
                    ZStack {
                        Circle()
                        .frame(width: 85.0, height: 85.0)
                        .foregroundColor(.blue)
                        
                        Button(action: {
                            if !self.showCamera {
                                self.showCamera = true
                            } else {
                                cameraController.capturePhoto()
                                self.showCamera = false
                                self.loadImage()
                            }
                        }) {
                            if self.showCamera {
                                Image(systemName: "circle.fill")
                                    .resizable()
                                    .frame(width: 70.0, height: 70.0)
                                    .foregroundColor(.white)
                            } else {
                                Image(systemName: "plus.circle.fill")
                                    .resizable()
                                    .frame(width: 70.0, height: 70.0)
                                    .foregroundColor(.white)
                            }
                        }
                    }.offset(x:0, y:-20)
                }
                Spacer()
                VStack {
                    Button(action: /*@START_MENU_TOKEN@*/{}/*@END_MENU_TOKEN@*/) {
                        Image(systemName: "person.circle.fill")
                            .imageScale(.large)
                            .foregroundColor(.white)
                    }
                    Text("Me")
                        .font(.caption)
                        .padding(5)
                        .foregroundColor(.white)
                }.padding()
                .frame(width: 80.0, height: 80.0)
            }.background(Color.blue)
        }
        .edgesIgnoringSafeArea(.bottom)
        .edgesIgnoringSafeArea(.top)
    }
    
    func loadImage() {
        guard let inputImage = inputImage else { return }
        image = Image(uiImage: inputImage)
    }
}

struct Home_Previews: PreviewProvider {
    static var previews: some View {
        Home()
    }
}
