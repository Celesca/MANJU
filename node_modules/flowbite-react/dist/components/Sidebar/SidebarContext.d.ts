import type { ThemingProps } from "../../types";
import type { SidebarTheme } from "./Sidebar";
export interface SidebarContextValue extends ThemingProps<SidebarTheme> {
    isCollapsed: boolean;
}
export declare const SidebarContext: import("react").Context<SidebarContextValue | undefined>;
export declare function useSidebarContext(): SidebarContextValue;
