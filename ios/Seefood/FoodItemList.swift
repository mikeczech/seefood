//
//  FoodItemList.swift
//  Seefood
//
//  Created by Mike Czech on 19.01.20.
//  Copyright © 2020 Mike Czech. All rights reserved.
//

import SwiftUI

struct FoodItemList: View {
    
    var foodItems: [FoodItem]
    
    var body: some View {
        List(foodItems) { foodItem in
            FoodItemRow(foodItem: foodItem)
        }
    }
}

struct FoodItemList_Previews: PreviewProvider {
    static var previews: some View {
        FoodItemList(foodItems: foodItemData)
    }
}
