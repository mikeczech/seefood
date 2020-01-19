//
//  NewEntryView.swift
//  Seefood
//
//  Created by Mike Czech on 17.01.20.
//  Copyright © 2020 Mike Czech. All rights reserved.
//

import SwiftUI


class CreateEntryModel: ObservableObject {
    
    var foodItems = [FoodItem]()
    @Published var isLoaded = false
        
    func fetchItems() {
        DispatchQueue.main.async {
            sleep(5) // TODO actually fetch items
            self.foodItems = foodItemData
            self.isLoaded = true
        }

    }
}

struct CreateEntry: View {
        
    @Binding var image: Image?
    @ObservedObject private var model = CreateEntryModel()
    
    var body: some View {
        GeometryReader { geometry in
            VStack {
                self.image?
                   .resizable()
                   .scaledToFill()
                   .frame(width: geometry.size.width, height: 400)
                   .clipped()
                if self.model.isLoaded {
                    FoodItemList(foodItems: self.model.foodItems)
                } else {
                    Spacer()
                    Text("Scanning image...")
                    ActivityIndicator(isAnimating: .constant(true), style: .large)
                }
                
                Spacer()
            }.onAppear {
                self.model.fetchItems()
            }
        }
        .edgesIgnoringSafeArea(.bottom)
        .edgesIgnoringSafeArea(.top)
    }
}

struct CreateEntryView_Previews: PreviewProvider {
    static var previews: some View {
        CreateEntry(image: .constant(Image("burger")))
    }
}
