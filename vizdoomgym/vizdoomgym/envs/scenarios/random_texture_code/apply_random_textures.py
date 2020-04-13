import omg
import sys
import random
import os

with open('animated_textures.txt') as f:
    ANIMATED_TEXTURES = f.read().split()

def change_textures(in_map, textures, index=None, animated_textures=None):
        
    #print textures
    
    # three options are: 
    #   all random; 
    #   all walls the same, ceil the same, floor the same; 
    #   all the same
    # probs = [1.,1.,1.]
    # r = random.uniform(0,probs[0] + probs[1] + probs[2])
    # if r < probs[0]:
    #     mode = 'all_rand'
    # elif r < probs[0] + probs[1]:
    #     mode = 'walls_ceil_floor'
    # else:
    #     mode = 'all_the_same'   

    mode = 'all_rand'
    # mode = 'walls_ceil_floor'
    # mode = 'all_the_same'
        
    map_editor = omg.MapEditor(in_map)
    
    apply_to_mid = True # this has to be False for lab22
    if mode == 'all_rand':
        for s in map_editor.sidedefs:
            up = random.choice(textures)
            low = random.choice(textures)
            if low in ANIMATED_TEXTURES:
                print('RED FLAG LOW: ', LOW)
            if up in ANIMATED_TEXTURES:
                print('RED FLAG UP: ', up)
            s.tx_up = up
            s.tx_low = low
            if apply_to_mid:
                mid = random.choice(textures)
                if mid in ANIMATED_TEXTURES:
                    print('RED FLAG MID: ', mid)
                s.tx_mid = mid
        for s in map_editor.sectors:
            floor = random.choice(textures)
            ceil = random.choice(textures)
            if floor in ANIMATED_TEXTURES:
                print('RED FLAG FLOOR: ', floor)
            if ceil in ANIMATED_TEXTURES:
                print('RED FLAG CEIL: ', ceil)
            s.tx_floor = floor
            s.tx_ceil = ceil
    elif mode == 'walls_ceil_floor':
        wall_tx = random.choice(textures)
        floor_tx = random.choice(textures)
        ceil_tx = random.choice(textures)
        for s in map_editor.sidedefs:
            s.tx_up = wall_tx
            s.tx_low = wall_tx
            if apply_to_mid:
                s.tx_mid = wall_tx
        for s in map_editor.sectors:
            s.tx_floor = floor_tx
            s.tx_ceil = ceil_tx
    elif mode == 'all_the_same':
        # all_tx = random.choice(textures)
        all_tx = textures[index]
        for s in map_editor.sidedefs:
            s.tx_up = all_tx
            s.tx_low = all_tx
            if apply_to_mid:
                s.tx_mid = all_tx
        for s in map_editor.sectors:
            s.tx_floor = all_tx
            s.tx_ceil = all_tx  
    else:
        raise Exception('Unknown mode', mode)
        
    out_map = map_editor.to_lumps()
    
    to_copy = ['BEHAVIOR']
    for t in to_copy:
        if t in in_map:
            out_map[t] = in_map[t]
    
    return out_map

if __name__ == '__main__':

    in_file = '../labyrinth_randtx.wad'
    out_file = '../labyrinth_randtx3.wad'
    textures_file = 'static_textures.txt'
    
    wad = omg.WAD(in_file)

    with open(textures_file) as f:
        textures = f.read().split() # list of textures

    for n in range(100):
        if n < 10:
            map_number = '0' + str(n)
        else:
            map_number = str(n)
        wad.maps['MAP{}'.format(map_number)] = change_textures(wad.maps['MAP01'], textures)
        print('-'*50)

    wad.to_file(out_file)






















