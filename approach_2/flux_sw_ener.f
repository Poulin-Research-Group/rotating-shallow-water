      subroutine euler_F(uvh, NLnm, ener, enst, Nx,Ny,params)
      implicit none
      integer Nx, Ny
      double precision uvh(3,0:Ny+1,0:Nx+1), ener, enst
      double precision NLnm(3,Ny,Nx)
      double precision params(6), dt

      dt = params(6)

cf2py intent(in) :: uvh, params
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: uvh, NLnm, ener, enst

      call flux_ener(uvh, NLnm, ener, enst, Nx,Ny,params)
      call Euler(uvh, dt, NLnm, Nx, Ny)

      end


c     calculate the flux and update the solution using AB2 step      
      subroutine ab2_F(uvh, NLn,NLnm, ener, enst, Nx,Ny,params)
      implicit none
      integer Nx, Ny
      double precision uvh(3,0:Ny+1,0:Nx+1), ener, enst
      double precision NLn(3,Ny,Nx), NLnm(3,Ny,Nx)
      double precision params(6), dt

      dt = params(6)

cf2py intent(in) :: uvh, NLnm, params
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: NLn, ener, enst, uvh

      call flux_ener(uvh, NLn, ener, enst, Nx,Ny,params)
      call AB2(uvh, dt, NLn, NLnm, Nx, Ny)

      end


c     calculate the flux and update the solution using AB3 step      
      subroutine ab3_F(uvh, NL,NLn,NLnm, ener, enst, Nx,Ny,params)
      implicit none
      integer Nx, Ny
      double precision uvh(3,0:Ny+1,0:Nx+1), ener, enst
      double precision NL(3,Ny,Nx), NLn(3,Ny,Nx), NLnm(3,Ny,Nx)
      double precision params(6), dt

      dt = params(6)

cf2py intent(in) :: uvh, NLn,NLnm, params
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: NL, ener, enst, uvh

      call flux_ener(uvh, NL, ener, enst, Nx,Ny,params)
      call AB3(uvh, dt, NL, NLn, NLnm, Nx, Ny)

      end


c     euler method for time stepping      
      subroutine Euler(uvh, dt, NLnm, Nx, Ny)
      implicit none
      integer Nx, Ny, i, j, k
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      double precision NLnm(3, Ny, Nx)
      double precision dt

cf2py intent(in,out) :: uvh
cf2py intent(in) :: dt, NLnm
cf2py intent(hide) :: Nx, Ny

      do k=1,3
        do i=1,Nx
          do j=1,Ny
            uvh(k,j,i) = uvh(k,j,i) + dt*NLnm(k,j,i)
          enddo
        enddo
      enddo

      ! Pad ghost cells
      do k=1,3
        do i=1,Nx
          uvh(k,0,i) = uvh(k,Ny,i)
          uvh(k,Ny+1,i) = uvh(k,1,i)
        enddo
        do j=0,Ny+1
          uvh(k,j,0) = uvh(k,j,Nx)
          uvh(k,j,Nx+1) = uvh(k,j,1)
        enddo
      enddo
      
      end


      subroutine AB2(uvh, dt, NLn, NLnm, Nx,Ny)
      implicit none
      integer Nx, Ny, i, j, k
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      double precision NLn(3, Nx, Ny), NLnm(3, Ny, Nx)
      double precision dt, dt2, dt3

      dt2 = 0.5*dt
      dt3 = 1.5*dt

cf2py intent(in,out) :: uvh
cf2py intent(in) :: dt, NLn, NLnm
cf2py intent(hide) :: Nx, Ny

      do k=1,3
         do i=1,Nx
            do j=1,Ny
               uvh(k,j,i) = uvh(k,j,i) + dt3*NLn(k,j,i)
     &                                 - dt2*NLnm(k,j,i)
            enddo
         enddo
      enddo

c     Pad ghost cells
      do k=1,3
        do i=1,Nx
          uvh(k,0,i) = uvh(k,Ny,i)
          uvh(k,Ny+1,i) = uvh(k,1,i)
        enddo
        do j=0,Ny+1
          uvh(k,j,0) = uvh(k,j,Nx)
          uvh(k,j,Nx+1) = uvh(k,j,1)
        enddo
      enddo
      
      end

      
      subroutine AB3(uvh, dt, NL, NLn, NLnm, Nx,Ny)
      implicit none
      integer Nx, Ny, i, j, k
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
         do i=1,Nx
            do j=1,Ny
               uvh(k,j,i) = uvh(k,j,i) + dt3*NL(k,j,i) - dt2*NLn(k,j,i)
     &                                 + dt1*NLnm(k,j,i)
            enddo
         enddo
      enddo

c     Pad ghost cells
      do k=1,3
        do i=1,Nx
          uvh(k,0,i) = uvh(k,Ny,i)
          uvh(k,Ny+1,i) = uvh(k,1,i)
        enddo
        do j=0,Ny+1
          uvh(k,j,0) = uvh(k,j,Nx)
          uvh(k,j,Nx+1) = uvh(k,j,1)
        enddo
      enddo
      
      end


c     calculate h, U, V, B, q --> flux, energy, enstrophy.
      subroutine flux_ener(uvh, flux, ener, enst, Nx, Ny, params)
      implicit none
      integer Nx, Ny, i, j
      double precision uvh(0:Ny+1,0:Nx+1,3)
      double precision flux(Ny,Nx,3), ener, enst

      double precision params(6), dx, dy, gp, f0, H0
      double precision h(0:Ny+1,0:Nx+1)
      double precision U(0:Ny+1,0:Nx+1), V(0:Ny+1,0:Nx+1)
      double precision B(0:Ny+1,0:Nx+1), q(0:Ny+1,0:Nx+1)

cf2py intent(in) :: uvh, params, inds
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: flux, ener, enst

      dx = params(1)
      dy = params(2)
      gp = params(3)
      f0 = params(4)
      H0 = params(5)

c     calc h
      do i=0,Nx+1
        do j=0,Ny+1
          h(j,i) = H0 + uvh(j,i,3)
        enddo
      enddo

c     calc U
      do i=1,Nx
        do j=1,Ny
          U(j,i) = 0.5*(h(j,i+1) + h(j,i)) * uvh(j,i,1)
        enddo
      enddo

      do i=1,Nx
        U(0,i) = U(Ny,i)
        U(Ny+1,i) = U(1,j)
      enddo
      do j=0,Ny+1
        U(j,0) = U(j,Nx)
        U(j,Nx+1) = U(j,1)
      enddo


c     calc V
      do i=1,Nx
        do j=1,Ny
          V(j,i) = 0.5*(h(j+1,i) + h(j,i)) * uvh(j,i,2)
        enddo
      enddo

      do i=1,Nx
        V(0,i) = V(Ny,i)
        V(Ny+1,i) = V(1,j)
      enddo
      do j=0,Ny+1
        V(j,0) = V(j,Nx)
        V(j,Nx+1) = V(j,1)
      enddo
      
      
c     calc B
      do i=1,Nx
        do j=1,Ny
          B(j,i) = gp*h(j,i)
     &        + 0.25*(uvh(j,i,1)*uvh(j,i,1) + uvh(j,i-1,1)*uvh(j,i-1,1))     
     &        + 0.25*(uvh(j,i,2)*uvh(j,i,2) + uvh(j-1,i,2)*uvh(j-1,i,2))
        enddo
      enddo

      do i=1,Nx
         B(0,i) = B(Ny,i)
         B(Ny+1,i) = B(1,i)
      enddo
      do j=0,Ny+1
         B(j,0) = B(j,Nx)
         B(j,Nx+1) = B(j,1)
      enddo
      
c     calc q
      do i=1,Nx
         do j=1,Ny
            q(j,i) = ((uvh(j,i+1,2) - uvh(j,i,2)) / dx 
     &              - (uvh(j+1,i,1) - uvh(j,i,1)) / dy +  f0)
     &           / (0.25*(h(j+1,i+1) + h(j+1,i) + h(j,i+1) + h(j,i)))
         enddo
      enddo

      do i=1,Nx
         q(0,i) = q(Ny,i)
         q(Ny+1,i) = q(1,i)
      enddo
      do j=0,Ny+1
         q(j,0) = q(j,Nx)
         q(j,Nx+1) = q(j,1)
      enddo

      do i=1,Nx
         do j=1,Ny
c     calc flux for u_t equation
            flux(j,i,1) = (q(j-1,i)*(V(j-1,i+1) + V(j-1,i)) 
     &                  +  q(j,i)*(V(j,i+1) + V(j,i)))*0.25 
     &                  - (B(j,i+1) - B(j,i)) / dx
            
c     calc flux for v_t equation
          flux(j,i,2) = -(q(j,i-1) * (U(j+1,i-1) + U(j,i-1))
     &                 +  q(j,i)   * (U(j+1,i)   + U(j,i)))*0.25 
     &                 - (B(j+1,i) - B(j,i)) / dy

c     calc flux for h_t equation
          flux(j,i,3) = (U(j,i-1) - U(j,i)) / dx
     &                + (V(j-1,i) - V(j,i)) / dy

         enddo
      enddo
      
      do i=1,Nx
         do j=1,Ny
            ener = ener + gp*h(j,i)*h(j,i) + 0.5*h(j,i)
     &           * (uvh(j,i,1)*uvh(j,i,1) + uvh(j,i-1,1)*uvh(j,i-1,1)
     &           +  uvh(j,i,2)*uvh(j,i,2) + uvh(j-1,i,2)*uvh(j-1,i,2))
            
            enst = enst + q(j,i)*q(j,i) 
     &           * (h(j+1,i+1) + h(j+1,i) + h(j,i+1) + h(j,i))
       enddo
      enddo

      end








