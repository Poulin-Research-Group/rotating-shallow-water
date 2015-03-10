      subroutine Euler_F(uvh, dt, NLnm, Nx, Ny3)
      implicit none

      double precision uvh(Ny3,Nx), NLnm(Ny3,Nx)
      double precision dt
c     Ny3 = 3 * Ny ... for convenience's sake.
      integer Nx, Ny3
      integer r, c

cf2py intent(in,out) :: uvh
cf2py intent(in) :: dt, NLnm
cf2py intent(hide) :: Nx, Ny3

      do r=1,Ny3
        do c=1,Nx
          uvh(r,c) = uvh(r,c) + dt*NLnm(r,c)
        enddo
      enddo

      end


      subroutine AB2_F(uvh, dt, NLn, NLnm, Nx, Ny3)
      implicit none

      double precision uvh(Ny3,Nx), NLn(Ny3,Nx), NLnm(Ny3,Nx)
      double precision dt
      integer Nx, Ny3
      integer r, c

cf2py intent(in,out) :: uvh
cf2py intent(in) :: dt, NLn, NLnm
cf2py intent(hide) :: Nx, Ny3

      do r=1,Ny3
        do c=1,Nx
          uvh(r,c) = uvh(r,c) + 0.5*dt*(3.0*NLn(r,c) - NLnm(r,c))
        enddo
      enddo

      end


      subroutine AB3_F(uvh, dt, NL, NLn, NLnm, Nx, Ny3)
      implicit none

      double precision uvh(Ny3,Nx), NL(Ny3,Nx)
      double precision NLn(Ny3,Nx), NLnm(Ny3,Nx)
      double precision dt
      integer Nx, Ny3
      integer r, c

cf2py intent(in,out) :: uvh
cf2py intent(in) :: dt, NL, NLn, NLnm
cf2py intent(hide) :: Nx, Ny3

      do r=1,Ny3
        do c=1,Nx
          uvh(r,c) = uvh(r,c) + dt/12.0*(23.0*NL(r,c) - 16.0*NLn(r,c)
     &                                   + 5.0*NLnm(r,c))
        enddo
      enddo

      end
