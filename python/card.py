#!/usr/bin/env python3
import sys
from math import sqrt, hypot
from random import randint
from array import array

class Vector3d:
    '''
    //Define a vector class with constructor and operator: 'v'
    struct v{
        f x,y,z;  // Vector has three float attributes.
        v operator+(v r){return v(x+r.x,y+r.y,z+r.z);} //Vector add
        v operator*(f r){return v(x*r,y*r,z*r);}       //Vector scaling
        f operator%(v r){return x*r.x+y*r.y+z*r.z;}    //Vector dot product
        v(){}                                  //Empty constructor
        v operator^(v r){return v(y*r.z-z*r.y,z*r.x-x*r.z,x*r.y-y*r.x);} //Cross-product
        v(f a,f b,f c){x=a;y=b;z=c;}            //Constructor
        v operator!(){return *this*(1 /sqrt(*this%*this));} // Used later for normalizing the vector
    };
    '''
    typecode = 'd'
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @classmethod
    def frombytes(cls, octets):
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(*memv)

    def __iter__(self):
        return (i for i in (self.x, self.y, self.z))

    def __repr__(self):
        class_name = type(self).__name__
        return '{}({!r}, {!r}, {!r})'.format(class_name, *self)

    def __str__(self):
        return str(tuple(self))

    def __bytes__(self):
        return (bytes([ord(self.typecode)]) +
                bytes(array(self.typecode, self)))

    def __eq__(self, other: Vector3d):
        return tuple(self) == tuple(other)

    def __abs__(self):
        return hypot(hypot(self.x, self.y), self.z)

    def __bool__(self):
        return bool(abs(self))

    def __add__(self, other: Vector3d):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scale: Vector3d):
        return Vector(self.x * scale, self.y * scale, self.z * scale)

    def __matmul__(self, other: Vector3d):
        return Vector(self.x * other.x, self.y * other.y, self.z * other.y)

    def _cross_product(self, other: Vector3d):
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def _normalisation(self):
        return self * (1 / sqrt(self.__dot__(self)))


'''
//The set of sphere positions describing the world.
//Those integers are in fact bit vectors.
i G[]={247570,280596,280600,249748,18578,18577,231184,16,16};

/*

16                    1    
16                    1    
231184   111    111   1    
18577       1  1   1  1   1
18578       1  1   1  1  1 
249748   1111  11111  1 1  
280600  1   1  1      11   
280596  1   1  1      1 1  
247570   1111   111   1  1 

*/
'''
G = [247570, 280596, 280600, 249748, 18578, 18577, 231184, 16, 16]


def random() -> float:
    '''
    // Random generator, return a float within range [0-1]
    f R(){return(f)rand()/RAND_MAX;}

    Sadly Python's `random.random` is `[0.0, 1.0)` so we have to
    use `random.randint` which is inclusive.
    '''
    return randint(0, 1000000) / 1000000.0


def intersection_test(o: Vector3d, d: Vector3d, t: float, n: Vector3d) -> int:
    '''
    //The intersection test for line [o,v].
    // Return 2 if a hit was found (and also return distance t and bouncing ray n).
    // Return 0 if no hit was found but ray goes upward
    // Return 1 if no hit was found but ray goes downward
    i T(v o,v d,f& t,v& n){ 
        t=1e9;
        i m=0;
        f p=-o.z/d.z;
        if(.01<p)
            t=p,n=v(0,0,1),m=1;

        //The world is encoded in G, with 9 lines and 19 columns
        for(i k=19;k--;)  //For each columns of objects
        for(i j=9;j--;)   //For each line on that columns
        
        if(G[j]&1<<k){ //For this line j, is there a sphere at column i ?
        
        // There is a sphere but does the ray hits it ?
            
        v p=o+v(-k,0,-j-4);
        f b=p%d,c=p%p-1,q=b*b-c;
        
        //Does the ray hit the sphere ?
        if(q>0){
            //It does, compute the distance camera-sphere
            f s=-b-sqrt(q);
                
            if(s<t && s>.01)
                // So far this is the minimum distance, save it. And also
                // compute the bouncing ray vector into 'n'  
                t=s,
                n=!(p+d*t),
                m=2;
            }
        }

        return m;
    }
    '''
    t = 10^9
    m = 0
    p = -o.z / d.z
    if .01 < p:
        t = p
        n = Vector3d(0., 0., 1.)
        m = 1
    
    for k in range(19, 0, -1):
        for j in range(9, 0, -1):
            if G[j] & 1<<k:
                # There is a sphere but does the ray hit it ?
                p = o + Vector3d(-k, 0.0, -j-4.0)
                b = p @ d
                c = p @ p - 1
                q = b * b - c

                # Does the ray hit the sphere
                if q > 0:
                    # It does, compute the distance camera<->sphere
                    s = -b - sqrt(q)
                    if s < t and s > .01:
                        # This is the minimum distance, save it. And alos
                        # compute the bouncing ray vector into `n`.
                        t = s
                        n = !(p + d + t)
                        m = 2

    return m

def sample(o: Vector3d, d: Vector3d) -> Vector3d:
    """
    // (S)ample the world and return the pixel color for
    // a ray passing by point o (Origin) and d (Direction)
    v S(v o,v d){
        f t;
        v n;

        //Search for an intersection ray Vs World.
        i m=T(o,d,t,n);


        if(!m) // m==0
        //No sphere found and the ray goes upward: Generate a sky color  
        return v(.7,.6,1)*pow(1-d.z,4);

        //A sphere was maybe hit.

        v h=o+d*t,                    // h = intersection coordinate
        l=!(v(9+R(),9+R(),16)+h*-1),  // 'l' = direction to light (with random delta for soft-shadows).
        r=d+n*(n%d*-2);               // r = The half-vector

        //Calculated the lambertian factor
        f b=l%n;

        //Calculate illumination factor (lambertian coefficient > 0 or in shadow)?
        if(b<0||T(h,l,t,n))
            b=0;

        // Calculate the color 'p' with diffuse and specular component 
        f p=pow(l%r*(b>0),99);

        if(m&1){   //m == 1
            h=h*.2; //No sphere was hit and the ray was going downward: Generate a floor color
            return((i)(ceil(h.x)+ceil(h.y))&1?v(3,1,1):v(3,3,3))*(b*.2+.1);
        }

        //m == 2 A sphere was hit. Cast an ray bouncing from the sphere surface.
        return v(p,p,p)+S(h,r)*.5; //Attenuate color by 50% since it is bouncing (* .5)
    }
    """
    # Search for an intersection ray vs. world
    m = intersection_test(o, d, t, n)

    if not m:
        # No sphere was found and the ray goes upward. Generate a sky colour
        return Vector3d(.7, .6, 1.) * (1-d.z ** 4)

    # A sphere was maybe hit
    h = o + d * t                                             # h = intersection co-ordinate
    l = !(Vector3d(9 + random(), 9 + random(), 16) + h * -1)  # l = direction to light (with random delta for soft-shadows)
    r = d + n * (n @ d * -2)                                  # r = The half-vector
    b = l @ n  # Calculate the lambertian factor

    if b < 0 or intersection_test(h, l, t, n):
        # Calculate the illumination factor (lambertian coefficient > 0 or in shadow) ?
        b = 0

    p = pow(l @ r * (b > 0), 99)  # Calculate the colour `p` with diffuse and specular component

    if m & 1:
        h *= .2  # No sphere was hit and the ray was going downward. Generate a floor colour
        return (ceil(h.x) + ceil(h.y)) & 1 ? Vector3d(3., 1., 1.) : Vector3d(3., 3., 3.) * (b * .2 + .1)
        
    # m == 2 -> No sphere was hit. Cast a ray bounding from the sphere surface.

    return Vector3d(p, p, p) + sample(h, r) * 0.5  # Attenuate colour by 50% since it is bouncing (* .5)


if __name__ == '__main__':
    """
    i main(){

        printf("P6 512 512 255 "); // The PPM Header is issued

        // The '!' are for normalizing each vectors with ! operator. 
        v g=!v(-6,-16,0),       // Camera direction
        a=!(v(0,0,1)^g)*.002, // Camera up vector...Seem Z is pointing up :/ WTF !
        b=!(g^a)*.002,        // The right vector, obtained via traditional cross-product
        c=(a+b)*-256+g;       // WTF ? See https://news.ycombinator.com/item?id=6425965 for more.

        for(i y=512;y--;)    //For each column
        for(i x=512;x--;){   //For each pixel in a line
        
        //Reuse the vector class to store not XYZ but a RGB pixel color
        v p(13,13,13);     // Default pixel color is almost pitch black
        
        //Cast 64 rays per pixel (For blur (stochastic sampling) and soft-shadows. 
        for(i r=64;r--;){ 
            
            // The delta to apply to the origin of the view (For Depth of View blur).
            v t=a*(R()-.5)*99+b*(R()-.5)*99; // A little bit of delta up/down and left/right 
                                            
            // Set the camera focal point v(17,16,8) and Cast the ray 
            // Accumulate the color returned in the p variable
            p=S(v(17,16,8)+t, //Ray Origin
                !(t*-1+(a*(R()+x)+b*(y+R())+c)*16) // Ray Direction with random deltas
                                                    // for stochastic sampling
                )*3.5+p; // +p for color accumulation
        }
        
        printf("%c%c%c",(i)p.x,(i)p.y,(i)p.z);
        
        }
    }
    """
    sys.stdout.write('P6 512 512 255 ')
    # The `!` are for normalising each vectors with `!` operator
    g = !Vector3d(-6., -16., 0.)
    a = !(Vector3d(0., 0., 1.) ^ g) * .002
    b = !(g ^ a) * .002
    c = (a + b) * -256 + g

    for y in range(512, 0, -1):
        for x in range(512, 0, -1):
            pixel = Vector3d(13., 13., 13.)  # Reuse the vector class to store RGB values
            # Cast 64 rays per pixel
            for r in range(64, 0, -1):
                t = a * (random() - .5) * 99 + b * (random() - .5) * 99
                p = sample(
                        Vector3d(17., 16., 8.) + t,  # Ray origin
                        # Ray direction with random deltas for stochastic sampling
                        !(t * -1 + (a * (random() + x) + b * (y + random()) + c ) * 16)
                    ) * 3.5 + p  # +p for colour accumulation

            sys.stdout.write('{!c}{!c}{!c}')
