//
//  NewEntryView.swift
//  Seefood
//
//  Created by Mike Czech on 17.01.20.
//  Copyright © 2020 Mike Czech. All rights reserved.
//

import SwiftUI

struct CreateEntryView: View {
        
    @Binding var image: Image?
    
    var body: some View {
        VStack {
            image?
                .resizable()
                .scaledToFill()
                .frame(height: 400)
                .clipped()
            
            Spacer()
        }
    }
}

struct CreateEntryView_Previews: PreviewProvider {
    static var previews: some View {
        CreateEntryView(image: .constant(nil))
    }
}
