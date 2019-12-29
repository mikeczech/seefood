//
//  Home.swift
//  Seefood
//
//  Created by Mike Czech on 29.12.19.
//  Copyright © 2019 Mike Czech. All rights reserved.
//

import SwiftUI

struct Home: View {
    var body: some View {
        VStack {
            Spacer()
            HStack(alignment: .top) {
                VStack {
                    Button(action: /*@START_MENU_TOKEN@*/{}/*@END_MENU_TOKEN@*/) {
                        Image(systemName: "house.fill")
                            .imageScale(.large)
                            .foregroundColor(.white)
                    }
                    Text("Home".uppercased())
                        .font(.caption)
                        .padding(5)
                        .foregroundColor(.white)
                }
                .padding()
                .frame(width: 80.0, height: 80.0)
                Spacer()
                VStack {
                    Button(action: /*@START_MENU_TOKEN@*/{}/*@END_MENU_TOKEN@*/) {
                        ZStack {
                            Circle()
                                .frame(width: 85.0, height: 85.0)
                                .foregroundColor(.pink)
                            Image(systemName: "plus.circle.fill")
                                .resizable()
                                .frame(width: 70.0, height: 70.0)
                                .foregroundColor(.white)
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
                    Text("me".uppercased())
                        .font(.caption)
                        .padding(5)
                        .foregroundColor(.white)
                }.padding()
                .frame(width: 80.0, height: 80.0)
            }.background(Color.pink)
        }.edgesIgnoringSafeArea(.bottom)
    }
}

struct Home_Previews: PreviewProvider {
    static var previews: some View {
        Home()
    }
}
