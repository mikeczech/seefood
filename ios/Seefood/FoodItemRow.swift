//
//  FoodItemRow.swift
//  Seefood
//
//  Created by Mike Czech on 19.01.20.
//  Copyright © 2020 Mike Czech. All rights reserved.
//

import SwiftUI

struct FoodItemRow: View {
    
    var foodItem: FoodItem
    
    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text(foodItem.category.rawValue)
                    .font(.title)
                Text("Calories ~\(foodItem.calories)")
                    .font(.body)
            }
            Spacer()
        }
        
    }
}

struct FoodItemRow_Previews: PreviewProvider {
    static var previews: some View {
        Group {
            FoodItemRow(foodItem: foodItemData[0])
            FoodItemRow(foodItem: foodItemData[1])
        }.previewLayout(.fixed(width:300, height: 70))
    }
}
