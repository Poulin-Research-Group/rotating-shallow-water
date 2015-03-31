      subroutine euler_F(uvh, NLnm, ener, enst, Nx,Ny,params,dims)
      implicit none
      integer dims(2), Nx, Ny
      double precision uvh(3,0:Ny+1,0:Nx+1), ener, enst
      double precision NLnm(3,Ny,Nx)
      double precision params(6), dt

      dt = params(6)

cf2py intent(in) :: uvh, params, dims
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: uvh, NLnm, ener, enst

      call flux_ener(uvh,NLnm,ener,enst,Nx,Ny,params)

      call Euler(uvh, dt, NLnm, Nx, Ny)

      end


c     calculate the flux and update the solution using AB2 step      
      subroutine ab2_F(uvh, NLn,NLnm, ener, enst, Nx,Ny,params,dims)
      implicit none
      integer dims(2), Nx, Ny
      double precision uvh(3,0:Ny+1,0:Nx+1), ener, enst
      double precision NLn(3,Ny,Nx), NLnm(3,Ny,Nx)
      double precision params(6), dt

      dt = params(6)

cf2py intent(in) :: uvh, NLnm, params, dims
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: uvh, NLn, ener, enst

      call flux_ener(uvh, NLn, ener, enst, Nx,Ny,params)
      call AB2(uvh, dt, NLn, NLnm, Nx, Ny)

      end


c     calculate the flux and update the solution using AB3 step      
      subroutine ab3_F(uvh, NL,NLn,NLnm, ener, enst, Nx,Ny,params,dims)
      implicit none
      integer dims(2), Nx, Ny
      double precision uvh(3,0:Ny+1,0:Nx+1), ener, enst
      double precision NL(3,Ny,Nx), NLn(3,Ny,Nx), NLnm(3,Ny,Nx)
      double precision params(6), dt

      dt = params(6)

cf2py intent(in) :: uvh, NLn,NLnm, params, dims
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: uvh, NL, ener, enst

      call flux_ener(uvh, NL, ener, enst, Nx,Ny,params)
      call AB3(uvh, dt, NL, NLn, NLnm, Nx, Ny)

      end


c     euler method for time stepping      
      subroutine Euler(uvh, dt, NLnm, Nx, Ny)
      implicit none
      integer Nx, Ny, c, r, k
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      double precision NLnm(3, Ny, Nx)
      double precision dt

cf2py intent(in,out) :: uvh
cf2py intent(in) :: dt, NLnm
cf2py intent(hide) :: Nx, Ny

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
      integer Nx, Ny, c, r, k
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      double precision NLn(3, Nx, Ny), NLnm(3, Ny, Nx)
      double precision dt, dt2, dt3

      dt2 = 0.5*dt
      dt3 = 1.5*dt

cf2py intent(in,out) :: uvh
cf2py intent(in) :: dt, NLn, NLnm
cf2py intent(hide) :: Nx, Ny

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
      integer Nx, Ny, c, r, k
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      double precision NL(3,Nx,Ny), NLn(3, Nx, Ny), NLnm(3, Ny, Nx)
      double precision dt, dt1, dt2, dt3

cf2py intent(in,out) :: uvh
cf2py intent(in) :: dt, NLn, NLnm
cf2py intent(hide) :: Nx, Ny

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
      integer Nx, Ny, c, r
      double precision uvh(3,0:Ny+1,0:Nx+1)
      double precision flux(3,Ny,Nx), ener, enst

      double precision params(6), dx, dy, gp, f0, H0
      double precision h(0:Ny+1,0:Nx+1)
      double precision U(0:Ny+1,0:Nx+1), V(0:Ny+1,0:Nx+1)
      double precision B(0:Ny+1,0:Nx+1), q(0:Ny+1,0:Nx+1)

cf2py intent(in) :: uvh, params
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: flux, ener, enst

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
      double precision A(0:Ny+1, 0:Nx+1)
      integer Nx, Ny, c, r

      intent(out) :: A

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
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      integer Nx, Ny, k, r, c

      intent(out) :: uvh

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