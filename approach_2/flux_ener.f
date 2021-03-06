      program sadourny
      implicit none
      
      double precision Lx, Ly, dx, dy
      double precision f0, gp, H0, hmax
      double precision params(6)

c     spatial parameters
      integer sc, Nx, Ny, Nt, Ny2, Ny3
      parameter (sc=1)
      parameter (Nx=128*sc)
      parameter (Ny=128*sc)
      parameter (Ny2=2*Ny)
      parameter (Ny3=3*Ny)

c     temporal parameters
      double precision dt, t0, tf
      parameter (t0 = 0.0)
      parameter (tf = 3600.0)
      parameter (dt = 5.0/sc)
      parameter (Nt = (tf - t0)/dt)

c     arrays
      double precision uvh(3,0:Ny+1,0:Nx+1)
      double precision NL(3,Ny,Nx), NLn(3,Ny,Nx), NLnm(3,Ny,Nx)
      double precision x(Nx), y(Ny), xs(Nx), ys(Ny)
      double precision ener(Nt), enst(Nt)
      
      integer dims(2)
      integer r, c, t, k

      ! write(6,*) "Nx, Ny, Nt", Nx, Ny, Nt
      
c     domain length 
      Lx = 200e3
      Ly = 200e3

c     compute grid
      dx = Lx/Nx
      dy = Ly/Ny

      do r=1,Ny
        y(r)  = -Ly/2 + (r-1)*dy
        ys(r) = -Ly/2 + (r-0.5)*dy
      enddo
      
      do c=1,Nx
        x(c)  = -Lx/2 + (c-1)*dx
        xs(c) = -Lx/2 + (c-0.5)*dx
      enddo

c     Physical parameters
      f0 = 1e-4
      gp = 9.81
      H0 = 500.

c     store parameters in a vector
      params(1) = dx
      params(2) = dy
      params(3) = f0
      params(4) = gp
      params(5) = H0
      params(6) = dt

c     Specify Initial Conditions
      hmax = 10.0
      do c=0,Nx+1
        do r=0,Ny+1
          uvh(1, r, c) = 0.0
          uvh(2, r, c) = 0.0
          uvh(3, r, c) = hmax*exp(-(x(c)**2 + y(r)**2)/(Lx/6.0)**2)
        enddo
      enddo

      
c     Euler solution
      call euler_F(uvh,NLnm,ener(1),enst(1),Nx,Ny,params,dims)
      
c     AB2 solution
      call AB2_F(uvh,NLn,NLnm,ener(2),enst(2),Nx,Ny,params,dims)

c     loop through time
      do t=3,Nt
c       AB3 solution
        call ab3_F(uvh,NL,NLn,NLnm,ener(t),enst(t),Nx,Ny,params,dims)

c       reset fluxes
        do k=1,3
          do c=1,Nx
            do r=1,Ny
              NLnm(k,r,c) = NLn(k,r,c)
              NLn(k,r,c)  = NL(k,r,c)
            enddo
          enddo
        enddo
         
      enddo

      stop
      end

      subroutine euler_F(uvh, NLnm, ener, enst, Nx,Ny,params,dims)
      implicit none
      intent(in) :: params, dims
      intent(out) :: NLnm, ener, enst
      integer dims(2), Nx, Ny
      double precision uvh(3,0:Ny+1,0:Nx+1), ener, enst
      double precision NLnm(3,Ny,Nx)
      double precision params(6), dt

      dt = params(6)


      call flux_ener(uvh,NLnm,ener,enst,Nx,Ny,params)

      call Euler(uvh, dt, NLnm, Nx, Ny)

      end


c     calculate the flux and update the solution using AB2 step      
      subroutine ab2_F(uvh, NLn,NLnm, ener, enst, Nx,Ny,params,dims)
      implicit none
      intent(in) :: NLnm, params, dims
      intent(out) :: NLn, ener, enst
      integer dims(2), Nx, Ny
      double precision uvh(3,0:Ny+1,0:Nx+1), ener, enst
      double precision NLn(3,Ny,Nx), NLnm(3,Ny,Nx)
      double precision params(6), dt

      dt = params(6)


      call flux_ener(uvh, NLn, ener, enst, Nx,Ny,params)
      call AB2(uvh, dt, NLn, NLnm, Nx, Ny)

      end


c     calculate the flux and update the solution using AB3 step      
      subroutine ab3_F(uvh, NL,NLn,NLnm, ener, enst, Nx,Ny,params,dims)
      implicit none
      intent(in) :: NLn,NLnm, params, dims
      intent(out) :: NL, ener, enst
      integer dims(2), Nx, Ny
      double precision uvh(3,0:Ny+1,0:Nx+1), ener, enst
      double precision NL(3,Ny,Nx), NLn(3,Ny,Nx), NLnm(3,Ny,Nx)
      double precision params(6), dt

      dt = params(6)


      call flux_ener(uvh, NL, ener, enst, Nx,Ny,params)
      call AB3(uvh, dt, NL, NLn, NLnm, Nx, Ny)

      end


c     euler method for time stepping      
      subroutine Euler(uvh, dt, NLnm, Nx, Ny)
      implicit none
      intent(in) :: dt, NLnm
      integer Nx, Ny, c, r, k
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      double precision NLnm(3, Ny, Nx)
      double precision dt


      do k=1,3
        do c=1,Nx
          do r=1,Ny
            uvh(k,r,c) = uvh(k,r,c) + dt*NLnm(k,r,c)
          enddo
        enddo
      enddo

      ! Pad ghost cells
      call ghostify_uvh(uvh, Nx, Ny)
      
      end


      subroutine AB2(uvh, dt, NLn, NLnm, Nx,Ny)
      implicit none
      intent(in) :: dt, NLn, NLnm
      integer Nx, Ny, c, r, k
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      double precision NLn(3, Nx, Ny), NLnm(3, Ny, Nx)
      double precision dt, dt2, dt3

      dt2 = 0.5*dt
      dt3 = 1.5*dt


      do k=1,3
         do c=1,Nx
            do r=1,Ny
               uvh(k,r,c) = uvh(k,r,c) + dt3*NLn(k,r,c)
     &                                 - dt2*NLnm(k,r,c)
            enddo
         enddo
      enddo

c     Pad ghost cells
      call ghostify_uvh(uvh, Nx, Ny)
      
      end

      
      subroutine AB3(uvh, dt, NL, NLn, NLnm, Nx,Ny)
      implicit none
      intent(in) :: dt, NLn, NLnm
      integer Nx, Ny, c, r, k
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      double precision NL(3,Nx,Ny), NLn(3, Nx, Ny), NLnm(3, Ny, Nx)
      double precision dt, dt1, dt2, dt3


      dt3 = 23./12.*dt
      dt2 = 16./12.*dt
      dt1 =  5./12.*dt

      do k=1,3
         do c=1,Nx
            do r=1,Ny
               uvh(k,r,c) = uvh(k,r,c) + dt3*NL(k,r,c) - dt2*NLn(k,r,c)
     &                                 + dt1*NLnm(k,r,c)
            enddo
         enddo
      enddo

c     Pad ghost cells
      call ghostify_uvh(uvh, Nx, Ny)
      
      end


c     calculate h, U, V, B, q --> flux, energy, enstrophy.
      subroutine flux_ener(uvh,flux,ener,enst, Nx,Ny,params)
      implicit none
      intent(in) :: uvh, params
      intent(out) :: flux, ener, enst
      integer Nx, Ny, c, r
      double precision uvh(3,0:Ny+1,0:Nx+1)
      double precision flux(3,Ny,Nx), ener, enst

      double precision params(6), dx, dy, gp, f0, H0
      double precision h(0:Ny+1,0:Nx+1)
      double precision U(0:Ny+1,0:Nx+1), V(0:Ny+1,0:Nx+1)
      double precision B(0:Ny+1,0:Nx+1), q(0:Ny+1,0:Nx+1)

      ! REMOVE THIS ===============================================================================
      double precision max1, max2, max3


      dx = params(1)
      dy = params(2)
      f0 = params(3)
      gp = params(4)
      H0 = params(5)

c     calc h
      do c=0,Nx+1
        do r=0,Ny+1
          h(r,c) = H0 + uvh(3,r,c)
        enddo
      enddo

c     calc U
      do c=1,Nx
        do r=1,Ny
          U(r,c) = 0.5*(h(r,c+1) + h(r,c)) * uvh(1,r,c)
        enddo
      enddo
      call ghostify(U, Nx, Ny)

c     calc V
      do c=1,Nx
        do r=1,Ny
          V(r,c) = 0.5*(h(r+1,c) + h(r,c)) * uvh(2,r,c)
        enddo
      enddo
      call ghostify(V, Nx, Ny)
      
c     calc B
      do c=1,Nx
        do r=1,Ny
          B(r,c) = gp*h(r,c)
     &        + 0.25*(uvh(1,r,c)*uvh(1,r,c) + uvh(1,r,c-1)*uvh(1,r,c-1))     
     &        + 0.25*(uvh(2,r,c)*uvh(2,r,c) + uvh(2,r-1,c)*uvh(2,r-1,c))
        enddo
      enddo
      call ghostify(B, Nx, Ny)
     
c     calc q
      do c=1,Nx
         do r=1,Ny
            q(r,c) = ((uvh(2,r,c+1) - uvh(2,r,c)) / dx 
     &              - (uvh(1,r+1,c) - uvh(1,r,c)) / dy +  f0)
     &           / (0.25*(h(r+1,c+1) + h(r+1,c) + h(r,c+1) + h(r,c)))
         enddo
      enddo
      call ghostify(q, Nx, Ny)

c     calc fluxes
      do c=1,Nx
         do r=1,Ny
c     calc flux for u_t equation
            flux(1,r,c) = (q(r-1,c)*(V(r-1,c+1) + V(r-1,c)) 
     &                  +  q(r,c)*(V(r,c+1) + V(r,c)))*0.25 
     &                  - (B(r,c+1) - B(r,c)) / dx
            
c     calc flux for v_t equation
          flux(2,r,c) = -(q(r,c-1) * (U(r+1,c-1) + U(r,c-1))
     &                 +  q(r,c)   * (U(r+1,c)   + U(r,c)))*0.25 
     &                 - (B(r+1,c) - B(r,c)) / dy

c     calc flux for h_t equation
          flux(3,r,c) = (U(r,c-1) - U(r,c)) / dx
     &                + (V(r-1,c) - V(r,c)) / dy

         enddo
      enddo

c     calc energy and enstrophy
      do c=1,Nx
         do r=1,Ny
            ener = ener + gp*h(r,c)*h(r,c) + 0.5*h(r,c)
     &           * (uvh(1,r,c)*uvh(1,r,c) + uvh(1,r,c-1)*uvh(1,r,c-1)
     &           +  uvh(2,r,c)*uvh(2,r,c) + uvh(2,r-1,c)*uvh(2,r-1,c))

            enst = enst + q(r,c)*q(r,c) 
     &           * (h(r+1,c+1) + h(r+1,c) + h(r,c+1) + h(r,c))
       enddo
      enddo

      ener = 0.5*  ener / (Ny*Nx)
      enst = 0.125*enst / (Ny*Nx)

      end


      subroutine ghostify(A, Nx, Ny)
      implicit none
      intent(out) :: A
      double precision A(0:Ny+1, 0:Nx+1)
      integer Nx, Ny, c, r

      do c=1,Nx
        A(0,c) = A(Ny,c)
        A(Ny+1,c) = A(1,c)
      enddo
      do r=0,Ny+1
         A(r,0) = A(r,Nx)
         A(r,Nx+1) = A(r,1)
      enddo

      end

      subroutine ghostify_uvh(uvh, Nx, Ny)
      implicit none
      intent(out) :: uvh
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      integer Nx, Ny, k, r, c

      do k=1,3
        do c=1,Nx
          uvh(k,0,c) = uvh(k,Ny,c)
          uvh(k,Ny+1,c) = uvh(k,1,c)
        enddo
        do r=0,Ny+1
          uvh(k,r,0) = uvh(k,r,Nx)
          uvh(k,r,Nx+1) = uvh(k,r,1)
        enddo
      enddo

      end
