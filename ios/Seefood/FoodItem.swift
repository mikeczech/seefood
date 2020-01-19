//
//  FoodItem.swift
//  Seefood
//
//  Created by Mike Czech on 19.01.20.
//  Copyright © 2020 Mike Czech. All rights reserved.
//

import Foundation

struct FoodItem: Hashable, Codable, Identifiable {
    
    var id: Int
    var category: Category
    var calories: Int
    
    enum Category: String, CaseIterable, Codable, Hashable {
        case burger = "Burger"
        case fries = "French Fries"
        case drink = "Soft Drink"
    }
    
}
