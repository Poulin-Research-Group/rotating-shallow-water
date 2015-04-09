      program sadourny
      implicit none
      
      double precision Lx, Ly, dx, dy
      double precision f0, gp, H0, hmax
      double precision params(6)

      ! spatial parameters
      integer sc, Nx, Ny, Nt, Ny2, Ny3
      parameter (sc=1)
      parameter (Nx=128*sc)
      parameter (Ny=128*sc)
      parameter (Ny2=2*Ny)
      parameter (Ny3=3*Ny)

      ! temporal parameters
      double precision dt, t0, tf
      parameter (t0 = 0.0)
      parameter (tf = 3600.0)
      parameter (dt = 5.0/sc)
      parameter (Nt = (tf - t0)/dt)

      ! arrays
      double precision uvh(3,0:Ny+1,0:Nx+1)
      double precision NL(3,Ny,Nx), NLn(3,Ny,Nx), NLnm(3,Ny,Nx)
      double precision x(Nx), y(Ny), xs(Nx), ys(Ny)
      double precision ener(Nt), enst(Nt)
      
      integer dims(2)
      integer r, c, t, k

      ! write(6,*) "Nx, Ny, Nt", Nx, Ny, Nt
      
      ! domain length 
      Lx = 200e3
      Ly = 200e3

      ! compute grid
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

      ! Physical parameters
      f0 = 1e-4
      gp = 9.81
      H0 = 500.

      ! store parameters in a vector
      params(1) = dx
      params(2) = dy
      params(3) = f0
      params(4) = gp
      params(5) = H0
      params(6) = dt

      ! Specify Initial Conditions
      hmax = 10.0
      uvh(1, :, :) = 0.0
      uvh(2, :, :) = 0.0
      do c=0,Nx+1
        do r=0,Ny+1
          uvh(3, r, c) = hmax*exp(-(x(c)**2 + y(r)**2)/(Lx/6.0)**2)
        enddo
      enddo

      
      ! Euler solution
      call euler_F(uvh,NLnm,ener(1),enst(1),Nx,Ny,params,dims)
      
      ! AB2 solution
      call AB2_F(uvh,NLn,NLnm,ener(2),enst(2),Nx,Ny,params,dims)

      ! loop through time
      do t=3,Nt

        ! AB3 solution
        call ab3_F(uvh,NL,NLn,NLnm,ener(t),enst(t),Nx,Ny,params,dims)

        ! reset fluxes
        NLnm(:, :, :) = NLn(:, :, :)
        NLn(:,  :, :) = NL(:,  :, :)
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


      ! calculate the flux and update the solution using AB2 step      
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


      ! calculate the flux and update the solution using AB3 step      
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


      ! euler method for time stepping      
      subroutine Euler(uvh, dt, NLnm, Nx, Ny)
      implicit none
      intent(in) :: dt, NLnm
      integer Nx, Ny
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      double precision NLnm(3, Ny, Nx)
      double precision dt

      uvh(:, 1:Ny, 1:Nx) = uvh(:, 1:Ny, 1:Nx) + dt*NLnm

      ! Pad ghost cells
      call ghostify_uvh(uvh, Nx, Ny)
      
      end


      subroutine AB2(uvh, dt, NLn, NLnm, Nx,Ny)
      implicit none
      intent(in) :: dt, NLn, NLnm
      integer Nx, Ny
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      double precision NLn(3, Nx, Ny), NLnm(3, Ny, Nx)
      double precision dt, dt2, dt3

      dt2 = 0.5*dt
      dt3 = 1.5*dt

      uvh(:, 1:Ny, 1:Nx) = uvh(:, 1:Ny, 1:Nx) + dt3*NLn - dt2*NLnm

      ! Pad ghost cells
      call ghostify_uvh(uvh, Nx, Ny)
      
      end

      
      subroutine AB3(uvh, dt, NL, NLn, NLnm, Nx,Ny)
      implicit none
      intent(in) :: dt, NLn, NLnm
      integer Nx, Ny
      double precision uvh(3, 0:Ny+1, 0:Nx+1)
      double precision NL(3,Nx,Ny), NLn(3, Nx, Ny), NLnm(3, Ny, Nx)
      double precision dt, dt1, dt2, dt3


      dt3 = 23./12.*dt
      dt2 = 16./12.*dt
      dt1 =  5./12.*dt

      uvh(:,1:Ny,1:Nx) = uvh(:,1:Ny,1:Nx) + dt3*NL - dt2*NLn + dt1*NLnm

      ! Pad ghost cells
      call ghostify_uvh(uvh, Nx, Ny)
      
      end


      ! calculate h, U, V, B, q --> flux, energy, enstrophy.
      subroutine flux_ener(uvh,flux,ener,enst, Nx,Ny,params)
      implicit none
      intent(in) :: uvh, params
      intent(out) :: flux, ener, enst
      integer Nx, Ny
      double precision uvh(3,0:Ny+1,0:Nx+1)
      double precision flux(3,Ny,Nx), ener, enst

      double precision params(6), dx, dy, gp, f0, H0
      double precision h(0:Ny+1,0:Nx+1)
      double precision U(0:Ny+1,0:Nx+1), V(0:Ny+1,0:Nx+1)
      double precision B(0:Ny+1,0:Nx+1), q(0:Ny+1,0:Nx+1)


      dx = params(1)
      dy = params(2)
      f0 = params(3)
      gp = params(4)
      H0 = params(5)

      ! calc h
      h = H0 + uvh(3, :, :)

      ! calc U
      U(1:Ny,1:Nx) = 0.5*(h(1:Ny,2:Nx+1) + h(1:Ny,1:Nx)) * uvh(1,1:Ny,1:Nx)
      call ghostify(U, Nx, Ny)

      ! calc V
      V(1:Ny,1:Nx) = 0.5*(h(2:Ny+1, 1:Nx) + h(1:Ny,1:Nx)) * uvh(2,1:Ny,1:Nx)
      call ghostify(V, Nx, Ny)

      ! calc B
      B(1:Ny,1:Nx) = gp*h(1:Ny,1:Nx)                                       &
                   + 0.25*(uvh(1,1:Ny,1:Nx)**2 + uvh(1,1:Ny,0:Nx-1)**2     &
                   +       uvh(2,1:Ny,1:Nx)**2 + uvh(2,0:Ny-1,1:Nx)**2 )
      call ghostify(B, Nx, Ny)
      
      ! calc q
      q(1:Ny,1:Nx) = ((uvh(2,1:Ny,2:Nx+1) - uvh(2,1:Ny,1:Nx)) / dx             &
                   -  (uvh(1,2:Ny+1,1:Nx) - uvh(1,1:Ny,1:Nx)) / dy  +  f0)     &
                   / (0.25*(h(2:Ny+1,2:Nx+1) + h(2:Ny+1,1:Nx) + h(1:Ny,2:Nx+1) + h(1:Ny,1:Nx)))
      call ghostify(q, Nx, Ny)

      ! calc flux for u_t
      flux(1,1:Ny,1:Nx) = ( q(0:Ny-1,1:Nx) * (V(0:Ny-1,2:Nx+1) + V(0:Ny-1,1:Nx))       &
                        +   q(1:Ny,1:Nx)   * (V(1:Ny,2:Nx+1)   + V(1:Ny,1:Nx))) *0.25  &
                        - (B(1:Ny,2:Nx+1) - B(1:Ny,1:Nx)) / dx

      ! calc flux for v_t
      flux(2,1:Ny,1:Nx) = -(q(1:Ny,0:Nx-1) * (U(2:Ny,0:Nx-1) + U(1:Ny,0:Nx-1))         &
                        +   q(1:Ny,1:Nx)   * (U(2:Ny+1,1:Nx) + U(1:Ny,1:Nx))) *0.25    &
                        - (B(2:Ny+1,1:Nx) - B(1:Ny,1:Nx)) / dy

      ! calc flux for h_t
      flux(3,1:Ny,1:Nx) = (U(1:Ny,0:Nx-1) - U(1:Ny,1:Nx)) / dx  &
                        + (V(0:Ny-1,1:Nx) - V(1:Ny,1:Nx)) / dy
      

      ! calculating energy and enstrophy
      ener = SUM(gp*h(1:Ny,1:Nx)**2 + 0.5*h(1:Ny,1:Nx)          &
           * (uvh(1, 1:Ny, 1:Nx)**2 + uvh(1, 1:Ny, 0:Nx-1)**2   &
           +  uvh(2, 1:Ny, 1:Nx)**2 + uvh(2, 0:Ny-1, 1:Nx)**2) ) / (2*Ny*Nx)

      enst = SUM(q(1:Ny, 1:Nx)**2  &
           * (h(2:Ny+1, 2:Nx+1) + h(2:Ny+1, 1:Nx) + h(1:Ny, 2:Nx+1) + h(1:Ny,1:Nx)) ) / (8*Ny*Nx)

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
