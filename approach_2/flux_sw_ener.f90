      subroutine euler_F(uvh, NLnm, ener, enst, Nx,Ny,params)
      implicit none
      integer Nx, Ny
      double precision uvh(3,0:Ny+1,0:Nx+1), ener, enst
      double precision NLnm(3,Ny,Nx)
      double precision params(6), dt

      dt = params(6)

cf2py intent(in) :: uvh, params
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: NLnm, ener, enst, uvh

      call flux_ener(uvh,NLnm,ener,enst,Nx,Ny,params)

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
      integer Nx, Ny
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      double precision NLnm(3, Ny, Nx)
      double precision dt

cf2py intent(in,out) :: uvh
cf2py intent(in) :: dt, NLnm
cf2py intent(hide) :: Nx, Ny

      uvh(:, 1:Ny, 1:Nx) = uvh(:, 1:Ny, 1:Nx) + dt*NLnm

      ! Pad ghost cells
      call ghostify_uvh(uvh, Nx, Ny)
      
      end


      subroutine AB2(uvh, dt, NLn, NLnm, Nx,Ny)
      implicit none
      integer Nx, Ny
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      double precision NLn(3, Nx, Ny), NLnm(3, Ny, Nx)
      double precision dt, dt2, dt3

      dt2 = 0.5*dt
      dt3 = 1.5*dt

cf2py intent(in,out) :: uvh
cf2py intent(in) :: dt, NLn, NLnm
cf2py intent(hide) :: Nx, Ny

      uvh(:, 1:Ny, 1:Nx) = uvh(:, 1:Ny, 1:Nx) + dt3*NLn - dt2*NLnm

c     Pad ghost cells
      call ghostify_uvh(uvh, Nx, Ny)
      
      end

      
      subroutine AB3(uvh, dt, NL, NLn, NLnm, Nx,Ny)
      implicit none
      integer Nx, Ny
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      double precision NL(3,Nx,Ny), NLn(3, Nx, Ny), NLnm(3, Ny, Nx)
      double precision dt, dt1, dt2, dt3

cf2py intent(in,out) :: uvh
cf2py intent(in) :: dt, NLn, NLnm
cf2py intent(hide) :: Nx, Ny

      dt3 = 23./12.*dt
      dt2 = 16./12.*dt
      dt1 =  5./12.*dt

      uvh(:,1:Ny,1:Nx) = uvh(:,1:Ny,1:Nx) + dt3*NL - dt2*NLn + dt1*NLnm

c     Pad ghost cells
      call ghostify_uvh(uvh, Nx, Ny)
      
      end


c     calculate h, U, V, B, q --> flux, energy, enstrophy.
      subroutine flux_ener(uvh,flux,ener,enst, Nx,Ny,params)
      implicit none
      integer Nx, Ny
      double precision uvh(3,0:Ny+1,0:Nx+1)
      double precision flux(3,Ny,Nx), ener, enst

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
      h = H0 + uvh(3, :, :)

c     calc U
      U(1:Ny,1:Nx) = 0.5*(h(1:Ny,2:Nx+1) + h(1:Ny,1:Nx)) * uvh(1,1:Ny,1:Nx)
      call ghostify(U, Nx, Ny)

c     calc V
      V(1:Ny,1:Nx) = 0.5*(h(2:Ny+1, 1:Nx) + h(1:Ny,1:Nx)) * uvh(2,1:Ny,1:Nx)
      call ghostify(V, Nx, Ny)

c     calc B
      B(1:Ny,1:Nx) = gp*h(1:Ny,1:Nx)
     &  + 0.25*(uvh(1,1:Ny,1:Nx)**2 + uvh(1,1:Ny,0:Nx-1)**2
     &  +       uvh(2,1:Ny,1:Nx)**2 + uvh(2,0:Ny-1,1:Nx)**2 )
      call ghostify(B, Nx, Ny)
      
c     calc q
      q(1:Ny,1:Nx) = ((uvh(2,1:Ny,2:Nx+1) - uvh(2,1:Ny,1:Nx)) / dx
     &             -  (uvh(1,2:Ny+1,1:Nx) - uvh(1,1:Ny,1:Nx)) / dy  +  f0)
     &             / (0.25*(h(2:Ny+1,2:Nx+1) + h(2:Ny+1,1:Nx) + h(1:Ny,2:Nx+1) + h(1:Ny,1:Nx)))
      call ghostify(q, Nx, Ny)

c     calc flux for u_t
      flux(1,1:Ny,1:Nx) = ( q(0:Ny-1,1:Nx) * (V(0:Ny-1,2:Nx+1) + V(0:Ny-1,1:Nx)) 
     &                  +   q(1:Ny,1:Nx)   * (V(1:Ny,2:Nx+1)   + V(1:Ny,1:Nx))) *0.25
     &                  - (B(1:Ny,2:Nx+1) - B(1:Ny,1:Nx)) / dx

c     calc flux for v_t
      flux(2,1:Ny,1:Nx) = ( q(1:Ny,0:Nx-1) * (U(2:Ny,0:Nx-1) + U(1:Ny,0:Nx-1))
     &                  +   q(1:Ny,1:Nx)   * (U(2:Ny+1,1:Nx) + U(1:Ny,1:Nx))) *0.25
     &                  - (B(2:Ny+1,1:Nx) - B(1:Ny,1:Nx)) / dy

c     calc flux for h_t
      flux(3,1:Ny,1:Nx) = (U(1:Ny,0:Nx-1) - U(1:Ny,1:Nx)) / dx
     &                  + (V(0:Ny-1,1:Nx) - V(1:Ny,1:Nx)) / dy
      

      ! calculating energy and enstrophy
      ener = SUM(gp*h(1:Ny,1:Nx)**2 + 0.5*h(1:Ny,1:Nx)
     &     * (uvh(1, 1:Ny, 1:Nx)**2 + uvh(1, 1:Ny, 0:Nx-1)**2
     &     +  uvh(2, 1:Ny, 1:Nx)**2 + uvh(2, 0:Ny-1, 1:Nx)**2) ) / (2*Ny*Nx)

      enst = SUM(q(1:Ny, 1:Nx)**2
     &     * (h(2:Ny+1, 2:Nx+1) + h(2:Ny+1, 1:Nx) +  h(1:Ny, 2:Nx+1) + h(1:Ny,1:Nx)) ) / (8*Ny*Nx)

      end


      subroutine ghostify(A, Nx, Ny)
      double precision A(0:Ny+1, 0:Nx+1)
      integer Nx, Ny

      intent(out) :: A

      A(0, 1:Nx) = A(Ny, 1:Nx)
      A(Ny+1, 1:Nx) = A(1, 1:Nx)
      A(:, 0) = A(:, Nx)
      A(:, Nx+1) = A(:, 1)

      end

      subroutine ghostify_uvh(uvh, Nx, Ny)
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      integer Nx, Ny, k

      intent(out) :: uvh

      do k=1,3
        uvh(k, 0, :) = uvh(k, Ny, :)
        uvh(k, Ny+1, :) = uvh(k, 1, :)
        uvh(k, :, 0) = uvh(k, :, Nx)
        uvh(k, :, Nx+1) = uvh(k, :, 1)
      enddo

      end
