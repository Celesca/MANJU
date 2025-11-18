import { useState } from "react";
import { Disclosure, Menu } from "@headlessui/react";
import { Bars3Icon, XMarkIcon, BellIcon } from "@heroicons/react/24/outline";
import { NavLink } from "react-router-dom";

const navigation = [
  { name: "Home", href: "/" },
  { name: "About", href: "/about" },
  { name: "Pricing", href: "/pricing" },
  { name: "Contact", href: "/contact" },
];

function classNames(...classes) {
  return classes.filter(Boolean).join(" ");
}

export default function Navbar() {
  const [isLoggedIn, setIsLoggedIn] = useState(false); // State login

  return (
    <Disclosure as="nav" className="bg-gray-800">
      {({ open }) => (
        <>
          <div className="mx-auto max-w-7xl px-2 sm:px-6 lg:px-8">
            <div className="relative flex h-16 items-center justify-between">
              
              {/* Mobile menu button */}
              <div className="absolute inset-y-0 left-0 flex items-center sm:hidden">
                <Disclosure.Button className="inline-flex items-center justify-center rounded-md p-2 text-gray-400 hover:bg-white/5 hover:text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                  <span className="sr-only">Open main menu</span>
                  {open ? (
                    <XMarkIcon className="h-6 w-6" aria-hidden="true" />
                  ) : (
                    <Bars3Icon className="h-6 w-6" aria-hidden="true" />
                  )}
                </Disclosure.Button>
              </div>

              {/* Logo + Desktop menu */}
              <div className="flex flex-1 items-center justify-center sm:items-stretch sm:justify-start">
                <div className="flex flex-shrink-0 items-center">
                  <img
                    className="h-8 w-auto"
                    src="https://tailwindcss.com/plus-assets/img/logos/mark.svg?color=indigo&shade=500"
                    alt="Logo"
                  />
                </div>
                <div className="hidden sm:ml-6 sm:block">
                  <div className="flex space-x-4">
                    {navigation.map((item) => (
                      <NavLink
                        key={item.name}
                        to={item.href}
                        className={({ isActive }) =>
                          classNames(
                            isActive
                              ? "bg-gray-900 text-white"
                              : "text-gray-300 hover:bg-white/5 hover:text-white",
                            "rounded-md px-3 py-2 text-sm font-medium"
                          )
                        }
                      >
                        {item.name}
                      </NavLink>
                    ))}
                  </div>
                </div>
              </div>

              {/* Right section: Login / Sign Up หรือ Profile */}
              <div className="absolute inset-y-0 right-0 flex items-center pr-2 sm:static sm:inset-auto sm:ml-6 sm:pr-0 space-x-2">
                {isLoggedIn ? (
                  <>
                    {/* Notification */}
                    <button
                      type="button"
                      className="relative rounded-full p-1 text-gray-400 hover:text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    >
                      <span className="sr-only">View notifications</span>
                      <BellIcon className="h-6 w-6" aria-hidden="true" />
                    </button>

                    {/* Profile dropdown */}
                    <Menu as="div" className="relative ml-3">
                      <Menu.Button className="flex rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                        <span className="sr-only">Open user menu</span>
                        <img
                          className="h-8 w-8 rounded-full"
                          src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-1.2.1&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80"
                          alt=""
                        />
                      </Menu.Button>
                      <Menu.Items className="absolute right-0 mt-2 w-48 origin-top-right bg-white divide-y divide-gray-100 rounded-md shadow-lg focus:outline-none">
                        <Menu.Item>
                          {({ active }) => (
                            <a
                              className={`block px-4 py-2 text-sm text-gray-700 ${
                                active ? "bg-gray-100" : ""
                              }`}
                              href="#"
                            >
                              Your Profile
                            </a>
                          )}
                        </Menu.Item>
                        <Menu.Item>
                          {({ active }) => (
                            <a
                              className={`block px-4 py-2 text-sm text-gray-700 ${
                                active ? "bg-gray-100" : ""
                              }`}
                              href="#"
                            >
                              Settings
                            </a>
                          )}
                        </Menu.Item>
                        <Menu.Item>
                          {({ active }) => (
                            <button
                              onClick={() => setIsLoggedIn(false)}
                              className={`block w-full text-left px-4 py-2 text-sm text-gray-700 ${
                                active ? "bg-gray-100" : ""
                              }`}
                            >
                              Logout
                            </button>
                          )}
                        </Menu.Item>
                      </Menu.Items>
                    </Menu>
                  </>
                ) : (
                  <>
                    {/* Login / Sign Up */}
                    <NavLink
                      to="/login"
                      className="px-4 py-2 rounded-md text-gray-800 bg-white hover:bg-gray-100 text-sm font-medium"
                    >
                      Login
                    </NavLink>
                    <NavLink
                      to="/signup"
                      className="px-4 py-2 rounded-md text-white bg-indigo-600 hover:bg-indigo-700 text-sm font-medium"
                    >
                      Sign Up
                    </NavLink>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Mobile menu panel */}
          <Disclosure.Panel className="sm:hidden">
            <div className="space-y-1 px-2 pt-2 pb-3">
              {navigation.map((item) => (
                <NavLink
                  key={item.name}
                  to={item.href}
                  className={({ isActive }) =>
                    classNames(
                      isActive
                        ? "bg-gray-900 text-white"
                        : "text-gray-300 hover:bg-white/5 hover:text-white",
                      "block rounded-md px-3 py-2 text-base font-medium"
                    )
                  }
                >
                  {item.name}
                </NavLink>
              ))}
              {/* Mobile Login / Sign Up */}
              {!isLoggedIn && (
                <>
                  <NavLink
                    to="/login"
                    className="block rounded-md px-3 py-2 text-base font-medium text-gray-800 bg-white hover:bg-gray-100"
                  >
                    Login
                  </NavLink>
                  <NavLink
                    to="/signup"
                    className="block rounded-md px-3 py-2 text-base font-medium text-white bg-indigo-600 hover:bg-indigo-700"
                  >
                    Sign Up
                  </NavLink>
                </>
              )}
            </div>
          </Disclosure.Panel>
        </>
      )}
    </Disclosure>
  );
}
