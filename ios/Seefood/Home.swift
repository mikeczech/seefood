//
//  Home.swift
//  Seefood
//
//  Created by Mike Czech on 29.12.19.
//  Copyright © 2019 Mike Czech. All rights reserved.
//

import SwiftUI

struct Home: View {
    
    @State private var inputImage: Image?
    @State private var createEntry: Bool = false
        
    var body: some View {
        VStack {
            ZStack {
                CameraViewControllerRepresentable(inputImage: self.$inputImage, createEntry: self.$createEntry).sheet(isPresented: self.$createEntry) {
                    CreateEntryView(image: self.$inputImage)
                }
                VStack {
                    Spacer()
                    HStack(alignment: .top) {
                        VStack {
                            Button(action: {}) {
                                Image(systemName: "house.fill")
                                    .imageScale(.large)
                                    .foregroundColor(.black)
                            }
                            Text("Home")
                                .font(.caption)
                                .padding(5)
                                .foregroundColor(.black)
                        }
                        .padding()
                        .frame(width: 80.0, height: 80.0)
                        Spacer()
                        VStack {
                            Button(action: {}) {
                                Image(systemName: "waveform.path.ecg")
                                    .imageScale(.large)
                                    .foregroundColor(.black)
                            }
                            Text("Stats")
                                .font(.caption)
                                .padding(5)
                                .foregroundColor(.black)
                        }
                        .padding()
                        .frame(width: 80.0, height: 80.0)
                        Spacer()
                        VStack {
                            Button(action: /*@START_MENU_TOKEN@*/{}/*@END_MENU_TOKEN@*/) {
                                Image(systemName: "person.circle.fill")
                                    .imageScale(.large)
                                    .foregroundColor(.black)
                            }
                            Text("Me")
                                .font(.caption)
                                .padding(5)
                                .foregroundColor(.black)
                        }.padding()
                            .frame(width: 80.0, height: 80.0)
                    }
                    .background(Color.white)
                    .opacity(0.6)
                }
            }
            
        }
        .edgesIgnoringSafeArea(.bottom)
        .edgesIgnoringSafeArea(.top)
    }

}

struct Home_Previews: PreviewProvider {
    static var previews: some View {
        Home()
    }
}
