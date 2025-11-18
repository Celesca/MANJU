export interface SidebarItemContextValue {
    isInsideCollapse: boolean;
}
export declare const SidebarItemContext: import("react").Context<SidebarItemContextValue | undefined>;
export declare function useSidebarItemContext(): SidebarItemContextValue;
